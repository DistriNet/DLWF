from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
import numpy


def data():
    datapath = "./cw_datasets/tor_200w_2500tr_runs0_26.npz"
    maxlen = 2000
    minlen = 0
    v = 1
    max_features = 1
    test_split = 0.15

    print('Loading data {}'.format(datapath))
    npzfile = numpy.load(datapath)

    print("Read...")
    x = npzfile["data"]
    y = npzfile["labels"]
    npzfile.close()

    num_traces = {}
    if minlen > 0:
        if v:
            print("Filter minlen={}...".format(minlen))
            if traces:
                print("Filter traces={}...".format(traces))
        new_xs = []
        new_labels = []
        for x, y in zip(x, y):
            if y not in num_traces:
                num_traces[y] = 0
            count = num_traces[y]
            if traces:
                if count == traces:
                    continue
            if len(x) >= minlen:
                new_xs.append(x)
                new_labels.append(y)
                num_traces[y] = count + 1
        x = numpy.array(new_xs)
        y = numpy.array(new_labels)
        del new_xs, new_labels

    del num_traces

    if maxlen > 0:
        if len(x.shape) == 1 or x.shape[1] != maxlen:
            if v:
                print("Pad maxlen={}...".format(maxlen))
            x = sequence.pad_sequences(x, maxlen=maxlen, padding='post',
                                  truncating='post', dtype="float64")

    x = x.reshape(x.shape[0], x.shape[1], 1)

    print("Categorizing...")
    possible_y = list(set(y))

    dict_labels = {}
    i = 0
    for yy in possible_y:
        dict_labels[yy] = i
        i = i + 1
    print(dict_labels)

    new_y = []
    for yy in y:
        new_y.append(dict_labels[yy])

    y = np_utils.to_categorical(new_y)

    nb_instances = x.shape[0]
    nb_packets = x.shape[1]
    nb_classes = y.shape[1]
    nb_traces = int(nb_instances / nb_classes)

    print('Loaded data {} instances for {} classes: '
          '{} traces per class, {} packets per trace'.format(nb_instances,
                                                               nb_classes,
                                                               nb_traces,
                                                               nb_packets))
    seed = numpy.random.randint(1000)
    print('Shuffling data...')
    numpy.random.seed(seed)
    numpy.random.shuffle(x)
    numpy.random.seed(seed)
    numpy.random.shuffle(y)

    print('Splitting data...')
    x_train = numpy.array(x[:int(len(x) * (1 - test_split))])
    x_test = numpy.array(x[int(len(x) * (1 - test_split)):])

    del x

    y_train = numpy.array(y[:int(len(y) * (1 - test_split))])
    y_test = numpy.array(y[int(len(y) * (1 - test_split)):])

    del y

    return x_train, x_test, y_train, y_test, max_features, maxlen


'''
    cond_optimizer = conditional({{choice(['rmsprop', 'adadelta'])}})
    if cond_optimizer == 'rmsprop':
        optimizer = RMSprop(lr={{uniform(0.001, 0.003)}})
    elif cond_optimizer == 'sgd':
        optimizer = Adadelta(lr=1.0)
    else:
        optimizer = Adam(lr={{uniform(0.001, 0.003)}})
'''

'''
def lstm_model(x_train, x_test, y_train, y_test, max_features, maxlen):

    model = Sequential()

    model.add(LSTM(input_shape=(maxlen, max_features),
                   units={{choice([256, 128])}},
                   activation='sigmoid',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=True,
                   dropout={{uniform(0.2, 0.3)}}))

    model.add(LSTM(units={{choice([256, 128])}},
                   activation='sigmoid',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=False,
                   dropout={{uniform(0.2, 0.3)}}))

    model.add(Dense(units=nb_classes, activation='softmax'))

    optimizer = RMSprop(lr={{uniform(0.001, 0.0025)}})

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    for layer in model.layers:
        if "lstm" in layer.name:
            print(layer.name, layer.units, layer.activation, layer.dropout)
        else:
            print(layer.name)
    print(model.optimizer.lr.get_value())

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpointer = ModelCheckpoint(filepath='lstm_weights.hdf5',
                                   verbose=0,
                                   save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=10,
              validation_split=0.15,
              callbacks=[early_stopping, checkpointer])

    score, acc = model.evaluate(x_test, y_test, verbose=1)

    print('Test: loss {}, acc {}'.format(score, acc))

    ### Save the architecture
    json_string = model.to_json()

    ### Return the optimizer and the architecture
    return {'loss': -acc, 'status': STATUS_OK, 'model': json_string}
'''


def cnn_model(x_train, x_test, y_train, y_test, max_features, maxlen):

    ''' 
    kernel_size = {{choice([5, 20])}}
    filters = {{choice([32, 64])}}
    pool_size = {{choice([2, 4, 16])}}
    dropout = {{uniform(0.1, 0.3)}}
    batch_size = {{choice([32, 64, 128])}}
    lr = {{uniform(0.0009, 0.0025)}}
    '''

    kernel_size = 5
    filters = 32
    pool_size = 4
    dropout = 0.1
    batch_size = 128
    lr = 0.001
    ### {{choice([0.001, 0.002])}}
    dr = 'dr1'
    ### conditional({{choice(['dr1', 'dr2'])}})
    reg = 'no'
    ### conditional({{choice(['yes', 'no'])}})

    model = Sequential()

    model.add(Dropout(input_shape=(maxlen, max_features), rate=dropout))

    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=pool_size, padding='valid'))

    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    if dr == 'dr2':
        model.add(Dropout(rate=dropout))

    model.add(MaxPooling1D(pool_size=pool_size, padding='valid'))

    model.add(Flatten())

    if reg == 'yes':
        model.add(Dense(units=nb_classes, activation='softmax',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
    else:
        model.add(Dense(units=nb_classes, activation='softmax'))

    optimizer = RMSprop(lr=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())
    print("kernels {}, size {}, pool {}, lr {}, batch {}, dropout {}, reg {}, {}".format(filters, kernel_size, pool_size, lr, batch_size, dropout, reg, dr))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    checkpointer = ModelCheckpoint(filepath='lstm_weights.hdf5',
                                   verbose=0,
                                   save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=8,
              validation_split=0.15,
              callbacks=[early_stopping, checkpointer],
              verbose=1)

    score, acc = model.evaluate(x_test, y_test, verbose=1)

    print('Test: loss {}, acc {}'.format(score, acc))

    ### Save the architecture
    ### json_string = model.to_json()

    ### Return the optimizer and the architecture
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':

    trials = Trials()

    max_evals = 4

    best_run, best_model, space = optim.minimize(model=cnn_model,
                                                  data=data,
                                                  algo=tpe.suggest,
                                                  max_evals=max_evals,
                                                  trials=trials,
                                                  eval_space=True,
                                                  return_space=True)

    ### x_train, y_train, x_test, y_test, _, _ = data()
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("\n")

    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        ### print("Trial %s vals: %s" % (t, vals))
        print("Trial {}".format(t), eval_hyperopt_space(space, vals))
