import numpy as np
from keras.utils import np_utils
import keras.preprocessing.sequence as sq


def categorize(labels, dict_labels=None):
    possible_labels = list(set(labels))

    if not dict_labels:
        dict_labels = {}
        n = 0
        for label in possible_labels:
            dict_labels[label] = n
            n = n + 1

    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    new_labels = np_utils.to_categorical(new_labels)

    return new_labels


class DataGenerator(object):
    """Data generator for DLWF on Keras"""
    def __init__(self, batch_size=32, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, data, labels, indices):
        """Generates batches of samples"""
        nb_instances = data.shape[0]
        nb_classes = labels.shape[1]
        sample_shape = data[0].shape
        batch_data_shape = tuple([self.batch_size] + list(sample_shape))
        batch_label_shape = (self.batch_size, nb_classes)
        # Infinite loop
        while True:
            # Generate an exploration order
            #indices = np.arange(nb_instances)
            #if self.shuffle is True:
            #    np.random.shuffle(indices)

            # Generate batches
            imax = int(len(indices) / self.batch_size)
            for i in range(imax):
                # Form a batch
                x = np.empty(batch_data_shape)
                y = np.empty(batch_label_shape)
                for j, k in enumerate(indices[i * self.batch_size: (i + 1) * self.batch_size]):
                    x[j] = data[k]
                    y[j] = labels[k]
                if x.shape != batch_data_shape:
                    print(x.shape)
                    exit(0)
                yield x, y


def load_data(path, maxlen=None, minlen=0, traces=0, dnn_type=None, openw=False):
    """Load and shape the dataset"""
    npzfile = np.load(path)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()

    if minlen:
        num_traces = {}
        print("Filter on minlen={}\n".format(minlen))
        if traces:
            print("Filter on number of traces={}\n".format(traces))
        new_data = []
        new_labels = []
        for x, y in zip(data, labels):
            if y not in num_traces:
                num_traces[y] = 0
            count = num_traces[y]
            if traces:
                if count == traces:
                    continue
            if len(x) >= minlen:
                new_data.append(x)
                new_labels.append(y)
                num_traces[y] = count + 1
        data = np.array(new_data)
        labels = np.array(new_labels)
        del new_data, new_labels
        if not data.size:
            raise ValueError('After filtering, no sequence left.')
        del num_traces

    # Pad if traces are of various length or if their uniform length is not equal to maxlen
    if maxlen:
        if len(data.shape) == 1 or data.shape[1] != maxlen:
            print("Pad/trunc with maxlen={}".format(maxlen))
            #data = data[:, :maxlen]
            # Old way:
            data = sq.pad_sequences(data, maxlen=maxlen, padding='post', truncating='post', dtype="float64")

    if dnn_type == "lstm" or dnn_type == "cnn":
        data = data.reshape(data.shape[0], data.shape[1], 1)
    if type == "sdae" and len(data.shape) > 2:
        print("WEIRD! data.shape={}".format(data.shape))
        data = data.reshape(data.shape[0], data.shape[1])

    if not openw:
        print("Categorize")
        labels = categorize(labels)

    print("Data {}, labels {}".format(data.shape, labels.shape))

    return data, labels


def split_dataset(x, y, val_split=0.05, test_split=0.05):
    num = x.shape[0]
    split = int(num * (1 - test_split))

    x_test = np.array(x[split:])
    y_test = np.array(y[split:])

    num = x.shape[0] - x_test.shape[0]
    split = int(num * (1 - val_split))

    x_val = np.array(x[split:num])
    y_val = np.array(y[split:num])

    x_train = np.array(x[:split])
    y_train = np.array(y[:split])

    return x_train, y_train, x_val, y_val, x_test, y_test