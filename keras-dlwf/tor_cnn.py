import warnings
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.layers.recurrent import LSTM


def build_model(learn_params, nb_classes):
    input_length = learn_params["maxlen"]
    input_dim = learn_params["nb_features"]
    layers = learn_params["layers"]

    model = Sequential()

    maxlen = input_length
    max_features = input_dim

    if len(layers) == 0:
        raise("No layers")

    first_l = layers[0]
    rest_l = layers[1:]

    # First layer
    if first_l["name"] == 'dropout':
        model.add(Dropout(input_shape=(maxlen, max_features), rate=first_l.as_float('rate')))
    elif first_l["name"] == 'conv':
        model.add(Conv1D(filters=first_l.as_int('filters'),
                         kernel_size=first_l.as_int('kernel_size'),
                         padding='valid',
                         activation=first_l['activation'],
                         strides=first_l.as_int('stride')))

    # Middle layers (conv, dropout, pooling, dense, lstm.....)
    for l in rest_l:
        if l["name"] == 'maxpooling':
            model.add(MaxPooling1D(pool_size=l.as_int('pool_size'), padding='valid'))
        elif l["name"] == 'conv':
            model.add(Conv1D(filters=l.as_int('filters'),
                             kernel_size=l.as_int('kernel_size'),
                             padding='valid',
                             activation=l['activation'],
                             strides=l.as_int('stride')))
        elif l["name"] == 'dropout':
            model.add(Dropout(rate=l.as_float('rate')))
        elif l["name"] == 'lstm':
            model.add(LSTM(l.as_int('units')))
        elif l["name"] == 'flatten':
            model.add(Flatten())
        elif l["name"] == 'dense':
            if l.as_float('regularization') > 0.0:
                model.add(Dense(units=l.as_int('units'), activation=l['activation'],
                            kernel_regularizer=regularizers.l2(last_l.as_float('regularization')),
                            activity_regularizer=regularizers.l1(last_l.as_float('regularization'))))
            else:
                model.add(Dense(units=l.as_int('units'), activation=l['activation']))

    return model

