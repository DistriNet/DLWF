import argparse
import time
import os
import statistics
from datetime import datetime
import numpy as np
import math
from configobj import ConfigObj
from keras.utils import np_utils
import keras.preprocessing.sequence as sq
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import model_from_json
from hyperas.distributions import uniform
from data import load_data, split_dataset, DataGenerator
import tor_lstm
import tor_sdae
import tor_cnn


torconf = "tor.conf"
config = ConfigObj(torconf)
logfile = config['log']

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
#matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


def numpy_printopts(float_precision=6):
    float_formatter = lambda x: "%.{}f".format(float_precision) % x
    np.set_printoptions(formatter={'float_kind': float_formatter})  # precision


def curtime():
    return datetime.utcnow().strftime('%d.%m %H:%M:%S') #.%f')[:-3]


def log(id, s, dnn=None):
    print("> {}".format(s))
    if dnn is not None:
        l = open(dnn + "_log.out", "a")
    else:
        l = open(logfile, "a")
    l.write("ID{} {}>\t{}\n".format(id, curtime(), s))
    l.close()


def log_config(id):
    l = open("log_configs.out", "a")
    l.write("\nID{} {}\n".format(id, datetime.utcnow().strftime('%d.%m')))
    l.writelines(open(torconf, 'r').readlines())
    l.close()


def gen_id():
    return datetime.utcnow().strftime('%d%m_%H%M%S')


def plot_acc(acc, title, val_acc=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.ylim(0, 1)
    plt.plot(acc, label="Training", color='red')
    if val_acc is not None:
        plt.plot(val_acc, label="Validation", color='blue')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/acc{}.pdf'.format(imgdir, comment))
    plt.close()


def plot_loss(loss, title, val_loss=None, comment="", imgdir='imgdir'):
    plt.figure(figsize=(10, 4))
    plt.plot(loss, label="Training", color='purple')
    if val_loss is not None:
        plt.ylim(0, 5)
        plt.plot(val_loss, label="Validation", color='green')
    plt.title(title, y=1.08)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode='expand', borderaxespad=0.)
    plt.yticks(np.arange(0, 5, 0.5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('{}/loss{}.pdf'.format(imgdir, comment))
    plt.close()


def entropy(probs):
    e = 0.0
    for prob in probs:
        if prob == 0.0:
            continue
        e += prob * math.log(prob) #ln
    return -e


def log_results(id, fname, predicted, nb_classes, dnn=None, labels=None, resdir='resdir'): #, fname, res_dir=config.res_dir):
    r = open("{}/{}.csv".format(resdir, fname), "w")

    r.write("correct;label;predicted;predicted_prob;entropy")

    for cl in range(0, nb_classes):
        r.write(";prob_{}".format(cl))

    r.write("\n")

    class_result = np.argmax(predicted, axis=-1)

    acc = 0.0
    num = len(predicted)
    for res in range(0, num):
        predicted_label = int(class_result[res])
        prob = predicted[res][predicted_label]
        ent = entropy(predicted[res])

        if labels is not None:
            label = int(np.argmax(labels[res], axis=-1))
            correct = int(label == predicted_label)
            acc += correct
        else:
            label = "-"
            correct = "-"
        r.write("{};{};{};{:.4f};{:.4f}".format(correct, label, predicted_label, prob, ent))
        for cl in range(0, nb_classes):
            r.write(";{:.4f}".format(predicted[res][cl]))
        r.write("\n")
    r.close()

    if labels is not None:
        acc /= num
        log(id, "Accuracy:\t{}".format(acc), dnn)

    log(id, "Predictions saved to {}".format(fname), dnn)


def predict(id, model, data, batch_size=1, steps=0, gen=False):
    if gen:
        score = model.evaluate_generator(data, steps)
        predicted = model.predict_generator(data, steps)
    else:
        (x, y) = data
        score = model.evaluate(x, y, batch_size=batch_size, verbose=1)
        predicted = model.predict(x)

    test_loss = round(score[0], 4)
    test_acc = round(score[1], 4)
    log(id, "Test loss(entropy):\t{}".format(test_loss))
    log(id, "Test accuracy:\t{}".format(test_acc))

    return predicted, test_acc, test_loss


def run(id, cv, data_params, learn_params, model=None):

    nb_instances = data_params["nb_instances"]
    nb_classes = data_params["nb_classes"]
    nb_traces = data_params["nb_traces"]

    print('Building model...')

    if model is None:
        if learn_params['dnn_type'] == "lstm":
            model = tor_lstm.build_model(learn_params, nb_classes)
        elif learn_params['dnn_type'] == "sdae":
            model = tor_sdae.build_model(learn_params, nb_classes,
                                         data_params['train_gen'], data_params['val_gen'],
                                         steps=(learn_params['train_steps'], learn_params['val_steps']),
                                         pre_train=False)
        else:  # elif learn_params['dnn_type'] == "cnn":
            model = tor_cnn.build_model(learn_params, nb_classes)

    metrics = ['accuracy']

    optimizer = None
    if learn_params['optimizer'] == "sgd":
        optimizer = SGD(lr=learn_params['lr'],
                        decay=learn_params['decay'],
                        momentum=0.9,
                        nesterov=True)
    elif learn_params['optimizer'] == "adam":
        optimizer = Adam(lr=learn_params['lr'],
                         decay=learn_params['decay'])
    else:  # elif learn_params['optimizer'] == "rmsprop":
        optimizer = RMSprop(lr=learn_params['lr'],
                            decay=learn_params['decay'])

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    print(model.summary())

    start = time.time()
    # Train model on dataset
    history = model.fit_generator(generator=data_params['train_gen'],
                                  steps_per_epoch=learn_params['train_steps'],
                                  validation_data=data_params['val_gen'],
                                  validation_steps=learn_params['val_steps'],
                                  epochs=learn_params['epochs'])

    log(id, 'Training took {:.2f} sec'.format(time.time() - start))

    #print("LEN PREDICTED {}".format(len(predicted)))
    #log_results(id, "res_{}_{}".format(dnn, id), predicted, y_test, nb_classes, resdir=config['resdir'])

    tr_loss = round(history.history['loss'][-1], 4)
    tr_acc = round(history.history['acc'][-1], 4)

    '''
    plot_loss(history.history['loss'],
              title="ID{} {}w {}tr".format(id, nb_classes, nb_traces),
              val_loss=history.history['val_loss'],
              comment="{}_{}".format(id, cv),
              imgdir=config['imgdir'])

    plot_acc(history.history['acc'],
             title="ID{} {}w {}tr".format(id, nb_classes, nb_traces),
             val_acc=history.history['val_acc'],
             comment="{}_{}".format(id, cv),
             imgdir=config['imgdir'])
    '''
    return tr_loss, tr_acc, model


def parse_model_name(model_path):
    name = os.path.basename(model_path)
    return name.split("_")[0] + "_" + name.split("_")[1], name.split("_")[2]


def load_model(model_path):
    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + ".h5")
    return loaded_model


def eval_main():
    model_path = config['model_path']
    id, dnn = parse_model_name(model_path)
    print("Loading {} model {} from disk...".format(dnn, model_path))
    model = load_model(model_path)

    test_data = config['test_data']
    optimizer = config[dnn]['optimizer']
    decay = config[dnn][optimizer].as_float('decay')
    wang = config.as_bool('wang')
    maxlen = config[dnn].as_int('maxlen')
    openw = config.as_bool('openw')

    if openw:
        print("Open World")

    print('Loading data... ')
    x, y = load_tor_data(test_data,
                         type=dnn,
                         usecols=None,
                         maxlen=maxlen,
                         wang=wang,
                         openw=openw)

    nb_instances = x.shape[0]
    nb_packets = x.shape[1]

    log(id, 'Loaded data {} test instances {} packets long'.format(nb_instances, nb_packets), dnn)

    metrics = ['accuracy']

    if optimizer == "sgd":
        optimizer = SGD()
    elif optimizer == "adam":
        optimizer = Adam()
    elif optimizer == "rmsprop":
        optimizer = RMSprop()

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    start = time.time()
    if openw:
        predicted = model.predict(x, verbose=1)
    else:
        predicted, _, _ = predict(id, model, (x, y))
    log(id, 'Test took {:.2f} sec'.format(time.time() - start), dnn)

    save_res = True
    if save_res:
        if not openw:
            nb_classes = y.shape[1]
            log_results(id, "res_{}x{}_{}_{}".format(nb_classes, nb_instances, id, dnn), predicted, nb_classes, dnn=dnn, labels=y, resdir="results_" + dnn)
        else:
            log_results(id, "res_open{}_{}_{}".format(nb_instances, id, dnn), predicted, 200, dnn=dnn, resdir="results_" + dnn)


def main(save=False, wtime=False):
    id = gen_id()

    datapath = config['datapath']
    cross_val = config.as_int('cv')
    traces = config.as_int('traces')
    dnn = config['dnn']
    seed = config.as_int('seed')
    minlen = config.as_int('minlen')

    nb_epochs = config[dnn].as_int('nb_epochs')
    batch_size = config[dnn].as_int('batch_size')
    val_split = config[dnn].as_float('val_split')
    test_split = config[dnn].as_float('test_split')
    optimizer = config[dnn]['optimizer']
    nb_layers = config[dnn].as_int('nb_layers')
    layers = [config[dnn][str(x)] for x in range(1, nb_layers + 1)]
    lr = config[dnn][optimizer].as_float('lr')
    decay = config[dnn][optimizer].as_float('decay')
    maxlen = config[dnn].as_int('maxlen')


    nb_features = 1

    log_config(id)

    start = time.time()
    print('Loading data {}... '.format(datapath))
    data, labels = load_data(datapath,
                             minlen=minlen,
                             maxlen=maxlen,
                             traces=traces,
                             dnn_type=dnn)
    end = time.time()

    print("Took {:.2f} sec to load.".format(end - start))

    nb_instances = data.shape[0]
    nb_cells = data.shape[1]
    nb_classes = labels.shape[1]
    nb_traces = int(nb_instances / nb_classes)

    log(id, 'Loaded data {} instances for {} classes: '
            '{} traces per class, {} Tor cells per trace'.format(nb_instances,
                                                                 nb_classes,
                                                                 nb_traces,
                                                                 nb_cells))

    # CROSS-VALIDATION
    log_exp_name = "experiments.csv"
    if os.path.isfile(log_exp_name):
        log_exp = open(log_exp_name, "a")
    else:
        log_exp = open(log_exp_name, "a")
        log_exp.write(";".join(["ID", "w", "tr", "tr length", "DNN", "N layers", "lr",
                                "epochs", "tr loss", "tr acc", "cv", "test loss", "test acc", "std acc"]) + "\n")

    all_test_acc = []
    all_test_loss = []
    best_tr_loss = None
    best_tr_acc = None
    best_test_loss = 10
    #best_test_acc = None
    best_model = None

    ID = id
    indices = np.arange(nb_instances)
    for cv in range(1, cross_val + 1):
        model = None

        #seed = seed + np.random.randint(1000)
        #print("Shuffling data")
        #np.random.shuffle(indices)
        #start = time.time()
        #data = np.take(data, axis=0, indices=indices)
        #labels = np.take(labels, axis=0, indices=indices)
        #end = time.time()
        #print("Took {:.2f} sec to shuffle.".format(end - start))

        #start = time.time()
        # Split the dataset into training, validation and test sets
        #x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(data, labels, val_split, test_split)
        #end = time.time()
        #print("Took {:.2f} sec to split.".format(end - start))

        # del data
        # del labels

        # Generators
        # start = time.time()
        # train_gen = DataGenerator(batch_size=batch_size).generate(x_train, y_train)
        # val_gen = DataGenerator(batch_size=batch_size).generate(x_val, y_val)
        # end = time.time()
        # print("Took {:.2f} sec to make the generators.".format(end - start))

        #----alternative way---

        np.random.shuffle(indices)
        num = nb_instances

        split = int(num * (1 - test_split))
        ind_test = np.array(indices[split:])

        num = indices.shape[0] - ind_test.shape[0]
        split = int(num * (1 - val_split))

        ind_val = np.array(indices[split:num])
        ind_train = np.array(indices[:split])

        # Generators
        start = time.time()
        train_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_train)
        val_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_val)
        test_gen = DataGenerator(batch_size=1).generate(data, labels, ind_test)
        end = time.time()
        print("Took {:.2f} sec to make the generators.".format(end - start))

        data_params = {'train_gen': train_gen,
                       'val_gen': val_gen,
                       # 'test_data': (x_test, y_test),
                       'nb_instances': nb_instances,
                       'nb_classes': nb_classes,
                       'nb_traces': nb_traces}

        learn_params = {'dnn_type': dnn,
                        'epochs': nb_epochs,
                        'train_steps': ind_train.shape[0] // batch_size,
                        'val_steps': ind_val.shape[0] // batch_size,
                        'nb_features': nb_features,
                        'batch_size': batch_size,
                        'optimizer': optimizer,
                        'nb_layers': nb_layers,
                        'layers': layers,
                        'lr': lr,
                        'decay': decay,
                        'maxlen': maxlen}

        log(id, "Experiment {}: seed {}".format(cv, seed))

        tr_loss, tr_acc, model = run(id, cv, data_params, learn_params, model)

        # Predict for test data
        #start = time.time()
        #x_test = np.take(data, axis=0, indices=ind_test)
        #y_test = np.take(labels, axis=0, indices=ind_test)
        #end = time.time()
        #print("Took {:.2f} sec to separate the test data.".format(end - start))

        start = time.time()
        #_, test_acc, test_loss = predict(id, model, (x_test, y_test), batch_size)
        _, test_acc, test_loss = predict(id, model, test_gen, steps=len(ind_test), gen=True)
        log(id, 'Test took {:.2f} sec'.format(time.time() - start))

        if cross_val == 1:
            best_model = model
            log_exp.write(";".join(list(map(lambda a: str(a), [id, nb_classes, nb_traces, nb_cells, dnn, nb_layers,
                                                               lr, nb_epochs, tr_loss, tr_acc,
                                                               "x", test_loss, test_acc, "x"]))) + "\n")
            break

        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        if test_loss < best_test_loss:
            best_model = model
            best_test_loss = test_loss
            #best_test_acc = test_acc
            best_tr_loss = tr_loss
            best_tr_acc = tr_acc

    id = ID

    if cross_val > 1:
        mean_loss = round(statistics.mean(all_test_loss), 4)
        std_loss = round(statistics.stdev(all_test_loss), 4)
        mean_acc = round(statistics.mean(all_test_acc), 4)
        std_acc = round(statistics.stdev(all_test_acc), 4)
        log(id, "CV Test loss: mean {}, std {}".format(mean_loss, std_loss))
        log(id, "CV Test accuracy: mean {}, std {}".format(mean_acc, std_acc))
        log_exp.write(";".join(list(map(lambda a: str(a), [id, nb_classes, nb_traces, nb_cells, dnn, nb_layers,
                                                           lr, nb_epochs, best_tr_loss, best_tr_acc,
                                                           cross_val, mean_loss, mean_acc, "+/-" + str(std_acc)]))) + "\n")

    if False:
        # TIME EXPERIMENT
        print('Test data (3 days)... ')
        x3d_test, y3d_test = load_data(
            "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_test3d_200w_100tr_runs20_21.npz",
            dnn_type=dnn,
            maxlen=maxlen)
        _, test_acc, test_loss = predict(id, best_model, (x3d_test, y3d_test), batch_size=1)
        print('Test data (10 days)... ')
        x10d_test, y10d_test = load_data(
            "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_test10d_200w_100tr_runs47_48.npz",
            dnn_type=dnn,
            maxlen=maxlen)
        _, test_acc, test_loss = predict(id, best_model, (x10d_test, y10d_test), batch_size=1)
        print('Test data (4 weeks)... ')
        x2w_test, y2w_test = load_data(
            "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_test2w_200w_100tr_runs2w0_2w1.npz",
            dnn_type=dnn,
            maxlen=maxlen)
        _, test_acc, test_loss = predict(id, best_model, (x2w_test, y2w_test), batch_size=1)
        print('Test data (6 weeks)... ')
        x4w_test, y4w_test = load_data(
            "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_test4w_200w_100tr_runs4w0_4w1.npz",
            dnn_type=dnn,
            maxlen=maxlen)
        _, test_acc, test_loss = predict(id, best_model, (x4w_test, y4w_test), batch_size=1)
        print('Test data (8 weeks)... ')
        x6w_test, y6w_test = load_data(
            "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_test6w_200w_100tr_runs6w0_6w1.npz",
            dnn_type=dnn,
            maxlen=maxlen)
        _, test_acc, test_loss = predict(id, best_model, (x6w_test, y6w_test), batch_size=1)


    if save:
        # serialize model to JSON
        model_json = best_model.to_json()
        with open("models/{}_{}.json".format(id, dnn), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        best_model.save_weights("models/{}_{}.h5".format(id, dnn))
        print("Saved model {}_{} to disk".format(id, dnn))

        # clean up
        del best_model


#Main Run Thread
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

    parser.add_argument('--save', '-s',
                        action="store_true",
                        help='save the trained model (for cv: the best one)')
    parser.add_argument('--wtime', '-wt',
                        action="store_true",
                        help='time experiment: test time datasets')
    parser.add_argument('--eval', '-e',
                        action="store_true",
                        help='test the model')

    args = parser.parse_args()

    if not args.eval:
        main(args.save, args.wtime)
    else:
        eval_main()
