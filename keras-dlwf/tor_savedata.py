import numpy as np
import glob
import os
import pandas as pd
import sys
import h5py
from enum import Enum
import itertools
import math
import targets
import argparse
from closed_worlds import CW_900


class Feature(Enum):
    timestamp = 0
    length = 1
    direction = 2
    wang = 3


TS = "timestamp"
DIR = "direction"
LEN = "length"
NUM = "num_cells"
ACK = "ack"
WANG = "wang"


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


def expand2(dir, num_cells):
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded


def expand(num_cells):
    dir = int(math.copysign(1, num_cells))
    num_cells = abs(num_cells)
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded


def to_wang(trace, maxlen):
    #new_trace = []
    #for x in np.nditer(trace):
    #    new_trace.append(expand(x))
    #trace = np.hstack(new_trace)

    # truncate to maxlen
    trace = trace[:maxlen]
    # pad to maxlen
    trace = np.pad(trace, (0, maxlen - trace.size), 'constant')
    trace.astype(np.int8)
    return trace


def parse_target(filename):
    #return filename.split("_")[1:-1][0]
    return filename.split("-")[0]


def smart_shuffle(lists, list=False):
    if list:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x is not None and x.any()]
    else:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x]


#def save_data(dirpath, savepath, towang=True, h5=False):
    #fnames = glob.glob(os.path.join(dirpath, "*/*/*.csv"))
def save_data(txtname, savepath, maxlen, traces, openw, towang=True, h5=False): #, type)
    """Saves the Tor dataset.
    # Arguments
        txtname: path to txt file with list of files with traffic traces
        savepath: path to file where to save the data
        compress: compress the data
        h5: save in h5 format as well
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    features = ["direction", "num_cells"]
    #if towang:
    #    nb_features = 1
    #else:
    #   nb_features = len(features)

    fnames = [x.strip() for x in open(txtname, "r").readlines()]
    nb_traces = len(fnames)

    print("Saving dataset with {} traces".format(nb_traces))

    labels = np.empty([nb_traces], dtype=np.dtype(object))
    #if type == "lstm":
    #    data = np.empty([nb_traces, maxlen, nb_features], dtype=np.int16)

    data = np.empty([nb_traces, maxlen], dtype=np.int16)

    num_labels = {}
    i = 0
    for f in fnames:
        label = parse_target(os.path.basename(f))
        if traces:
            if label not in num_labels:
                num_labels[label] = 1
            else:
                count = num_labels[label]
                if count == traces:
                    continue
                else:
                    num_labels[label] = count + 1
        #try:
        df = pd.read_csv(f, nrows=10000, sep="\t", dtype=None, usecols=[0,1], header=None)
        #except:
        #    continue
        if df.empty:
            print("empty", f)
            continue
        if towang:
            #values = to_wang(df["direction"] * df["num_cells"], maxlen)
            values = to_wang(df[df.columns[1]]/1443, maxlen)
        else:
            print("WRONG")
            values = df.values

        data[i] = values
        labels[i] = label
        i += 1

    labels = np.array(labels)
    nb_classes = len(set(labels))
    data = np.array(data)

    # Save in npz
    savepath = savepath + '.npz'
    np.savez_compressed(savepath, data=data, labels=labels)
    print('Saved a dataset with {} traces for {} websites to {}'.format(data.shape[0], nb_classes, savepath))

    if h5:
        # Save hdf5
        h5f = h5py.File('{}.h5'.format(savepath), 'w')
        print(data)
        h5f.create_dataset('data', data=data)
        h5f.create_dataset('labels', data=labels)
        h5f.close()


if __name__ == "__main__":
    #URLs = bad_targets()
    #print(URLs)
    parser = argparse.ArgumentParser(description='Save files from txt to npz dataset')

    parser.add_argument('--file', '-f',
                        type=str,
                        help='txt file with list of files')
    parser.add_argument('--out', '-o',
                        type=str,
                        help='output dataset name')
    #parser.add_argument('--type', '-t',
    #                    type=str,
    #                    default="sdae",
    #                    help='type of DNN: \'sdae\' or \'lstm\'')
    parser.add_argument('--maxlen', '-m',
                        type=int,
                        default=5000,
                        help='max amount of features')
    parser.add_argument('--traces', '-t',
                        type=int,
                        default=0,
                        help='amount of traces')
    parser.add_argument('--openw', '-ow',
                        action="store_true",
                        help='open world dataset')

    args = parser.parse_args()
    save_data(args.file, args.out, args.maxlen, args.traces, args.openw)
