#PyMongo's API supports all of the features of MongoDB's map/reduce engine.
from pymongo import MongoClient
import re
import threading
import os
import errno
import csv
import time
import config
import multiprocessing as mp
from functools import partial
from itertools import repeat
import multiprocessing as mp
# from bson.code import Code
import targets


def bad_targets(exc_dupl=True, exc_unknown=True, exc_similar=False, exc_empty=True, exc_alm_empty=False):
    exclude = []
    if exc_dupl:
        for dupls in targets.duplicates:
            exclude += dupls[1:]
    if exc_unknown:
        exclude += targets.unknown_error
    if exc_similar:
        exclude += targets.similar
    if exc_empty:
        exclude += targets.empty
    if exc_alm_empty:
        exclude += targets.almost_empty
    return exclude


'''
def analyze_ex(stderr):
    return stderr


def log_ex(stderr, count, fail):
    with open("exs.txt", 'a') as f:
        f.write(str(count) + ". ")
        if fail:
            f.write("FAIL")
        else:
            f.write("No fail")
        f.write("\n" + stderr + "\n")


def log_fr(fr):
    with open("exs.txt", 'a') as f:
        f.write("Fail reason:\t" + fr + "\n")


def process_exs(colname):
    db = get_db(connect())
    print("Processing exceptions in the database...\n")
    start = time.time()
    ex_list = []
    count = 1

    for i in range(0, config.runs + 1):
        colname = config.colname_base + str(i).zfill(3)
        print("Loading failures from collection " + colname + " ...\n")
        collection = db[colname].find({"stderr":{ "$ne": ""}})
        exceptions = [entry for entry in collection]
        collection.close()
        print("Found " + str(len(exceptions)) + " exceptions.\n")

        for ex in exceptions:
            if 'stderr' not in ex:
                print("WEIRD")
                continue
            #analyze_ex(dict_list, ex['stderr'])
            stderr = ex['stderr']

            if stderr not in ex_list:
                log_ex(stderr, count, 'output' in ex and 'fail' in ex['output'] and ex['output']['fail'] is True)

                count += 1
                ex_list.append(stderr)

    print("\nUnique exceptions: " + str(len(ex_list)))
    end = time.time()

    print("Done, took " + str((end - start)) + " sec.\n")


def get_csv(cursor, visits_per_batch, outdir):
    for entry in cursor:
        url, visit = parse_id(str(entry['_id']))
        batch = str(int(visit / visits_per_batch))
        visit = str(visit % visits_per_batch)
        if 'parsed_dump' in entry:
            data = entry['parsed_dump']
            csvpath = gen_csvpath(outdir, batch, url, visit)
            write_to_csv(csvpath, data)
        else:
            print("WEIRD")
'''


def parse_id(id):
    split_id = id.split("_batch:")
    url = split_id[0].split("http://")[1]
    visit = int(re.findall(r'\d+', split_id[1])[0])
    return url, visit


def gen_csvpath(outdir, colname, url, visit):
    batch = str(int(colname.split("_")[-1]))
    dirpath = os.path.join(outdir, colname, url)

    try:
        os.makedirs(dirpath)  # , exist_ok=True)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    filename = '_'.join((batch, url, visit)) + ".csv"
    path = os.path.join(dirpath, filename)
    return path


def write_to_csv(csvpath, data):
    with open(csvpath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=config.sep, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["packet_index","timestamp", "length", "direction", "num_cells", "ack"])
        for pckt in data:
            writer.writerow([str(pckt['packet_index']),
                             str(pckt['timestamp']),
                             str(pckt['length']),
                             str(pckt['direction']),
                             str(pckt['num_cells']),
                             str(pckt['ack'])])


def connect():
    URI = 'mongodb://{}:{}@{}:{}/{}?readPreference=primary'.format(config.user,
                                                                   config.pswd,
                                                                   config.host,
                                                                   config.port,
                                                                   config.dbname)
    return MongoClient(URI)


def get_db(client): 
    db = client[config.dbname]
    return db


def save_csvs(exclude=[], colname="tor_run", old=False, openw=False):
    print("!", openw)
    db = get_db(connect())

    print("Loading collection {}...".format(colname))
    if old:
        query = db[colname].find({"status": {"$exists": False}, "_num_parsed_dump_entries": {"$gt": 0}},
                                  {"parsed_dump": 1,
                                  "_num_parsed_dump_entries": 1})
    elif openw:
        query = db[colname].find({"status": {"$exists": False}, "_num_parsed_dump_entries": {"$gt": 0}},
                                 {"parsed_dump": 1,
                                  "_num_parsed_dump_entries": 1,
                                  "domain": 1})
    else:
        query = db[colname].find({"status": {"$exists": True}, "_num_parsed_dump_entries": {"$gt": 0},
                                  "$or": [{"domain":"facebook.com"}, {"domain": "messenger.com"}, {"domain": "subscene.com"}, {"domain": "nicovideo.jp"}]},
                                 {"parsed_dump": 1,
                                  "_num_parsed_dump_entries": 1,
                                  "config": 1,
                                  "domain": 1})
    for entry in query:
        if old:
            url, visit = parse_id(str(entry['_id']))
            visit = str(visit)
        elif openw:
            url = str(entry['domain'])
            visit = "0"
        else:
            url = str(entry['domain'])
            visit = str(entry['config']['batch'])

        if url in exclude:
            continue

        if url not in ["facebook.com","messenger.com","subscene.com","nicovideo.jp"]:
            continue

        data = entry['parsed_dump']
        csvpath = gen_csvpath(config.outdir, colname, url, visit)
        write_to_csv(csvpath, data)

    print("{} is done.".format(colname))

    '''
    cursors = collection.parallel_scan(2)

    threads = [
        threading.Thread(target=get_csv, args=(cursor, config.visits_per_batch, config.outdir,))
        for cursor in cursors]

    if True:
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
    '''


def star(args):
    return args[0](*args[1:])


def gen_cols_time():
    for x in ["tor_run_v1_6weeks_later_000", "tor_run_v1_6weeks_later_001","tor_run_v1_6weeks_later_002"]:
        yield x


def gen_cols():
    for i in range(0, config.runs + 1):
        colname = config.colname_base + str(i).zfill(3)
        yield colname


if __name__ == '__main__':
    print("Saving data to " + config.outdir)
    proc_pool = mp.Pool(4)#mp.cpu_count())
    gcols = gen_cols_time() #gen_cols()
    exclude = bad_targets()
    part = partial(save_csvs, exclude, openw=config.openw)
    tasks = zip(repeat(part), gcols)
    proc_pool.map(star, tasks)

    #results = lambda l: [item for results in l for item in results]
    #unique_results = set(results)
    #print("\nUnique exceptions: " + str(len(unique_results)))
