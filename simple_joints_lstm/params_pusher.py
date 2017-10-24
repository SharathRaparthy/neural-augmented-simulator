HIDDEN_NODES = 128
LSTM_LAYERS = 3
EXPERIMENT = 1
# CUDA = False
CUDA = True
EPOCHS = 1
DATASET_PATH_REL = "/data/lisa/data/sim2real/"
DATASET_PATH = DATASET_PATH_REL + "mujoco_data2_pusher.h5"
MODEL_PATH = "./trained_models/simple_lstm_pusher{}_v1_{}l_{}.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
MODEL_PATH_BEST = "./trained_models/simple_lstm_pusher{}_v1_{}l_{}_best.pt".format(
    EXPERIMENT,
    LSTM_LAYERS,
    HIDDEN_NODES
)
TRAIN = True
CONTINUE = True
