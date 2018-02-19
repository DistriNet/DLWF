from configobj import ConfigObj
config = ConfigObj()
config.filename = "tor.conf"


cnn = {
    'nb_epochs': 10,
    'maxlen': 3000,
    'features': [2],
    'batch_size': 256,
    'val_split': 0.05,
    'test_split': 0.05,
    'optimizer': "rmsprop",# ""sgd"  # "adam"
    'nb_layers': 6,
    '1': {
        'name': 'dropout',
        'rate': 0.1
    },
    '2': {
        'name': 'conv',
        'filters': 32,
        'kernel_size': 5,
        'activation': 'relu',
        'stride': 1
    },
    '3': {
        'name': 'maxpooling',
        'pool_size': 4
    },
    '4': {
        'name': 'conv',
        'filters': 32,
        'kernel_size': 5,
        'activation': 'relu',
        'stride': 1
    },
    '5': {
        'name': 'maxpooling',
        'pool_size': 4
    },
    '6': {
        'name': 'dense',
        'regularization': 0.0
    },
    'rmsprop': {
        'lr': 0.0010053829131721615, #0.0028053829131721615
        'decay': 0.0
        }
}


lstm = {
    'nb_epochs': 50,
    'maxlen': 150,
    'features': [2],
    'batch_size': 256,
    'val_split': 0.15,
    'test_split': 0.15,
    'optimizer': "rmsprop",# ""sgd"  # "adam"
    'nb_layers': 2,
    '1': {
        'units': 128,
        'dropout': 0.22244615886559121,
        'activation': 'tanh',
        'rec_activation': 'hard_sigmoid'
    },
    '2': {
        'units': 128,
        'dropout': 0.20857652372682717,
        'activation': 'tanh',
        'rec_activation': 'hard_sigmoid'
    },
    'rmsprop': {
        'lr': 0.0010053829131721615, #0.0028053829131721615
        'decay': 0.0
        },
    'sgd': {
        'lr': 0.001,
        'momentum': 0.9,
        'decay': 0.0,
        'nesterov': True
        },
    'adam': {
        'lr': 0.001,
        'decay': 0.0
        }
}
'''
'2': {
    'units': 50,
    'dropout': 0.3,
    'activation': 'tanh',
    'rec_activation': 'hard_sigmoid'
}
'''

sdae = {
    'nb_epochs': 30,
    'maxlen': 5000,
    'features': [2],
    'batch_size': 32,
    'val_split': 0.05,
    'test_split': 0.05,
    'optimizer': 'sgd',
    'nb_layers': 3, # + 1 for classification
    'sgd': {
        'lr': 0.001,
        'momentum': 0.9,
        'decay': 0.0,
        'nesterov': True
        },
    '1': {
        'in_dim': 5000,
        'out_dim': 1000,
        'epochs': 50,
        'batch_size': 128,
        'dropout': 0.2,
        'optimizer': 'sgd',
        'sgd': {
            'lr': 0.001,
            'momentum': 0.9,
            'decay': 0.0
            },
        'enc_activation': 'tanh',
        'dec_activation': 'linear'
        },
    #'2': {
    #    'in_dim': 1000,
    #    'out_dim': 700,
    #    'epochs': 10,
    #    'batch_size': 128,
    #    'dropout': 0.2,
    #    'optimizer': 'sgd',
    #    'sgd': {
    #        'lr': 0.001,
    #        'momentum': 0.9,
    #        'decay': 0.0
    #        },
    #    'enc_activation': 'tanh',
    #    'dec_activation': 'linear'
    #    },
    '2': {
        'in_dim': 700,
        'out_dim': 500,
        'epochs': 10,
        'batch_size': 128,
        'dropout': 0.2,
        'optimizer': 'sgd',
        'sgd': {
            'lr': 0.001,
            'momentum': 0.9,
            'decay': 0.0
            },
        'enc_activation': 'tanh',
        'dec_activation': 'linear'
        },
    '3': {
        'in_dim': 500,
        'out_dim': 300,
        'epochs': 10,
        'batch_size': 128,
        'dropout': 0.2,
        'optimizer': 'sgd',
        'sgd': {
            'lr': 0.001,
            'momentum': 0.9,
            'decay': 0.0
            },
        'enc_activation': 'tanh',
        'dec_activation': 'linear'
        }
    #'3': {
    #    'in_dim': 500,
    #    'out_dim': 300,
    #    'epochs': 10,
    #    'batch_size': 128,
    #    'dropout': 0.2,
    #    'optimizer': 'sgd',
    #    'sgd': {
    #        'lr': 0.001,
    #        'momentum': 0.9,
    #        'decay': 0.0
    #        },
    #    'enc_activation': 'tanh',
    #    'dec_activation': 'linear'
    #    }
}

config['dnn'] = 'cnn'

#lstm_config = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/datasets_lstm/tor_100w_1000tr_runs000_010_lstm.npz"
lstm_config = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_900w_1500tr.npz"
sdae_config = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_train_200w_2000tr_runs0_19.npz"
#lstm_config#"/home/vera/deeplearn/dl-wf/src/keras-dlwf/tor_200w_2500tr_runs0_26.npz"
cnn_config = "cw100-1500-wtfpad-3.npz"
#"/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_time_train_200w_2000tr_runs0_19.npz"  #cw_datasets/tor_200w_2500tr_runs0_26.npz" 
#"/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/tor_900w_2500tr_runs0_29.npz" #500_shuffled.npz" #/tor_900w_2500tr_runs0_29.npz"#tor_100w_2000tr_runs0_30.npz"
#cnn_config = "/home/vera/tor/cw_datasets/tor_time_train_200w_2000tr_runs0_19.npz"

'''
tor_100w_2000tr_runs0_30.npz
tor_100w_1500tr_runs0_16.npz
tor_100w_1000tr_runs0_11.npz
tor_100w_500tr_runs0_6.npz
tor_100w_200tr_runs0_2.npz
tor_100w_100tr_runs0_2.npz

tor_100w_1000tr_runs0_10.npz
'''


config["traces"] = 1500#1920 #0
#config["test_data"] = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/tor_open_400000w_runs0_41.npz"
#config["test_data"] = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/tor_open_200w_2000tr_runs20_40.npz"
config["test_data"] = "/home/vera/tor/cw_datasets/tor_time_test6w_200w_100tr_runs6w0_6w1.npz"
'''
tor_time_test10d_200w_100tr_runs47_48.npz
../cw_datasets/tor_time_test3d_200w_100tr_runs20_21.npz
../cw_datasets/tor_time_test6w_200w_100tr_runs6w0_6w1.npz
../cw_datasets/tor_time_test2w_200w_100tr_runs2w0_2w1.npz
../cw_datasets/tor_time_test4w_200w_100tr_runs4w0_4w1.npz
'''
config["model_path"] = "/home/vera/tor/models/0808_155319_cnn"#1505_220909_sdae"#
config['openw'] = False

config['parts'] = False
config['parts_name'] = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/cw_datasets/parts/tor_part?_900w_500tr_runs0_29.npz"
#config['parts_name'] = "/home/vera/deeplearn/dl-wf/src/keras-dlwf/tor_part?_900w_500tr_runs0_29.npz"

config['lstm'] = lstm
config['sdae'] = sdae
config['cnn'] = cnn
config['seed'] = 18
config['cv'] = 1
config['minlen'] = 0


if config['dnn'] == 'sdae':
    config['imgdir'] = './images_sdae'
    config['resdir'] = './results_sdae'
    config['log'] = 'sdae_log.out'
    config['datapath'] = sdae_config
elif config['dnn'] == 'lstm':
    config['imgdir'] = './images_lstm'
    config['resdir'] = './results_lstm'
    config['log'] = 'lstm_log.out'
    config['datapath'] = lstm_config
else: #cnn
    config['imgdir'] = './images_cnn'
    config['resdir'] = './results_cnn'
    config['log'] = 'cnn_log.out'
    config['datapath'] = cnn_config

config.write()



'''
BEST LSTM

datapath = /home/vera/deeplearn/dl-wf/src/keras-dlwf/tor_100w_1000tr_runs0_11_sdae.npz
[lstm]
nb_epochs = 300
maxlen = 100
features = 2
batch_size = 256
val_split = 0.1
test_split = 0.2
optimizer = rmsprop
nb_layers = 2
[[1]]
units = 64
dropout = 0.1
activation = tanh
rec_activation = hard_sigmoid
[[2]]
units = 64
dropout = 0.1
activation = tanh
rec_activation = hard_sigmoid
[[sgd]]
lr = 0.1
momentum = 0.9
decay = 0.0
nesterov = True
[[adam]]
lr = 0.001
decay = 0.0
[[rmsprop]]
lr = 0.001
decay = 0.0

24-59 22:59:55.795 > Loaded data 100000 instances for 100 classes: 1000.0 traces per class, 150 packets per trace
25-39 03:39:26.503 > Training took 16768.40 sec
25-39 03:39:28.354 >
Test loss(entropy):     0.35559946675300597
25-39 03:39:28.354 > Test accuracy:     0.9163
25-39 03:39:50.504 > Test took 24.00 sec
'''
