#%matplotlib inline
import pandas as pd
import os
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from functools import partial
from pylab import rcParams
from functools import partial
from itertools import izip, repeat
import multiprocessing as mp
import argparse
import re

datapath1 = "/Users/vera/PycharmProjects/Mongo/data"
TS = "timestamp"
DIR = "direction"
LEN = "length"
NUM = "num_cells"
TD = "timedelta"
FIELDS = [TS, DIR, LEN, NUM]


def parse_target(filename):
    return filename.split("_")[1:-1]


def get_target_list(dirpath):
    target_list = []
    for _, _, files in os.walk(dirpath):
            for file in files:
                if file.endswith(".csv"):
                    target = parse_target(file)
                    if target not in target_list:
                        target_list.append(target)
    return target_list


# Compute the distribution of the feature(s) values over the whole dataset
# params: the list of features to analyze
def param_distr(datapath, params, target=None, sort=True):
    res_distr = {}
    for dirpath, _, files in os.walk(datapath):
        for file in files:
            if file.endswith(".csv") and (target is None or parse_target(file) == target):
                filepath = os.path.join(dirpath, file)
                #print(filepath)
                df = pd.read_csv(filepath, sep=";", index_col=0, dtype={TS: np.float64, LEN: np.int32, NUM: np.int32})
                if not set(FIELDS).issubset(df.columns):
                    raise Exception("Wrong format of the csv file header:\n" + filepath)
                for param in params:
                    counts = df[param].value_counts()
                    #print(counts)
                    if param not in res_distr:
                        res_distr[param] = counts
                    else:
                        res_distr[param] = res_distr[param].add(counts, fill_value=0).apply(np.int64)
    if sort:
        for param in params:
            res_distr[param].sort_values(axis=0, ascending=False, inplace=True)
    return res_distr


# Compute the evolution of the feature(s) values over the whole dataset
# params: the list of features to analyze
def param_evol(datapath, params, sep, target=None, timedelta=False):
    res_evols = {}
    sums = {}
    for param in params:
        sums[param] = {}
    for dirpath, _, files in os.walk(datapath):
        for file in files:
            if file.endswith(".csv") and (target is None or parse_target(file) == target):
                filepath = os.path.join(dirpath, file)
                #print(filepath)
                df = pd.read_csv(filepath, sep=sep, index_col=0, dtype={TS: np.float64, LEN: np.int32, NUM: np.int32})
                if not set(FIELDS).issubset(df.columns):
                    raise Exception("Wrong format of the csv file header:\n" + filepath)
                if df.empty or len(df) < 1:
                    #print("Empty " + filepath)
                    pass
                else:
                    ts = dt.fromtimestamp(df[TS].iloc[-1])
                    for param in params:
                        sums[param][ts] = np.array([df[param].sum()])
    for param in params:
        res_evols[param] = pd.DataFrame.from_items(sums[param].items(), orient='index', columns=[param])
        res_evols[param].sort_index(inplace=True)
        if timedelta:
            res_evols[param][TD] = res_evols[param].index
            res_evols[param][TD] = res_evols[param][TD].diff()
    return res_evols


def plot_evol(in_data, color='y', ylabel=None, label=None, resample=None, hours=1, kind='bar'):
    rcParams['figure.figsize'] = 17, 5
    ticklabels = None
    if resample is not None:
        resample_step = None
        tick_step = None
        dateformat = None
        if resample is '30min':
            resample_step = '30T'
            tick_step = 1
            dateformat='%d %b'
        if resample is 'day':
            resample_step = '1D'
            tick_step = 1
            dateformat='%d %b'
        elif resample is 'hour':
            resample_step = '{}H'.format(hours)
            tick_step = int(24/hours)
            dateformat='%d %b %Hhr'
        data = in_data.resample(resample_step).sum().dropna()
        ticklabels = [''] * len(data.index)
        ticklabels[::tick_step] = [item.strftime(dateformat) for item in data.index[::tick_step]]
    else:
        tick_step = 12
        ticklabels = [''] * len(data.index)
        ticklabels[::tick_step] = [item.strftime('%d %b %H:%M') for item in data.index[::tick_step]]

    prev_name = data.columns[0]
    data.rename(columns={prev_name:label}, inplace=True)

    # Plot a chart
    ax = data.plot(kind=kind, x=data.index, y=data.columns[0], color=color)
    ax.set_title('Tor traffic over crawling time')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate()
    plt.show()

    # Return the column name
    data.rename(columns={label:prev_name}, inplace=True)


def analyze(sep, target_list, csvfile):
	#print(csvfile)
	param_evol(in_dpath, [LEN], sep)
    evols[targ].rename(columns={prev_name:targ}, inplace=True)
    evols[targ].plot.hist(bins=100, color=colors[col_ind % len(colors)])
    #hist = evols[targ].hist()
    #hist.plot(kind='bar', color=colors[col_ind % len(colors)])
    img = os.path.join(datapath, targ + ".png")
    plt.savefig(img)


# Check format of a crawl csv filename <batch>_<target_url>_<visit>.csv
def is_crawl(name):
	return re.match("[0-9]+_[A-Za-z0-9-_.]+[.]+[A-Za-z0-9-_.]+_[0-9]+\.csv", name)


def gen_find_csv(in_dpath):
	for (dirpath, _, filenames) in os.walk(in_dpath):
		if not filenames:
			continue
		only_csvs = [os.path.join(dirpath, name) for name in filenames if is_crawl(name)]
		for csvfile in only_csvs:
			yield csvfile


def star(args):
	return args[0](*args[1:])


def main(in_dpath, sep, num_procs):
	proc_pool = mp.Pool(num_procs)
	print("INFO: Launching " + str(num_procs) + " processes.")

	target_list = get_target_list(datapath1)

	gcsvs = gen_find_csv(in_dpath, sep)
	part_analyze = partial(analyze, in_dpath, sep, target_list)
	tasks = izip(repeat(part_analyze), gcsvs)
	results = proc_pool.map(star, tasks)
	print(len(results))
	print(results[0])


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Tor data integrity analysis.')

	parser.add_argument('--in-dpath', '-i',
						default = '/Users/vera/PycharmProjects/Mongo/data/',
						type = str,
						help = 'path to the input directory (default = current directory)')

	parser.add_argument('--sep', '-s',
						type = str,
						default = ';',
						help = 'field separator of the input csv (default = \';\')')

	parser.add_argument('--num_procs', '-p',
						type = int,
						default = mp.cpu_count(),
						help = 'number of processes to spawn for processing (default = available in host)')

	args = parser.parse_args()
	main(**vars(args))
