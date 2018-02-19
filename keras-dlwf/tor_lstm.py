import warnings
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.models import Sequential
from hyperas.distributions import choice, uniform, conditional


#maxlen, nb_features, layers, nb_classes
def build_model(learn_params, nb_classes):
    input_length = learn_params["maxlen"]
    input_dim = learn_params["nb_features"]
    layers = learn_params["layers"]

    model = Sequential()
    # input_shape = (input_length, input_dim)
    # input_length = maxlen
    # input_dim = nb_features

    if len(layers) == 0:
        raise ("No layers")

    if len(layers) == 1:
        layer = layers[0]
        model.add(LSTM(input_shape=(input_length, input_dim),
                       #batch_input_shape=(batch_size, input_length, input_dim),
                       units=layer.as_int('units'),
                       activation=layer['activation'],
                       recurrent_activation=layer['rec_activation'],
                       return_sequences=False,
                       #stateful=True,
                       dropout=layer.as_float('dropout')))
        model.add(Dense(units=nb_classes, activation='softmax'))
        return model

    first_l = layers[0]
    last_l = layers[-1]
    middle_ls = layers[1:-1]
    #
    model.add(LSTM(input_shape=(input_length, input_dim),
                   #batch_input_shape=(batch_size, input_length, input_dim),
                   units=first_l.as_int('units'),
                   activation=first_l['activation'],
                   recurrent_activation=first_l['rec_activation'],
                   return_sequences=True,
                   #stateful=True,
                   dropout=first_l.as_float('dropout')))
    for l in middle_ls:
        model.add(LSTM(units=l.as_int('units'),
                       activation=l['activation'],
                       recurrent_activation=l['rec_activation'],
                       return_sequences=True,
                       #stateful=True,
                       dropout=l.as_float('dropout')))

    model.add(LSTM(units=last_l.as_int('units'),
                   activation=last_l['activation'],
                   recurrent_activation=last_l['rec_activation'],
                   return_sequences=False,
                   #stateful=True,
                   dropout=last_l.as_float('dropout')))

    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

