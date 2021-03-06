from CNN_BLSTM import *
from keras.optimizers import SGD, Nadam

EPOCHS = 10               # paper: 80
DROPOUT = 0.68            # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 275     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended


classes = ['state', 'area_land_sq_mi', 'density_sq_mi', 'pop', 'area_percentage', 'district', 'seat_wl', 'area_total_sq_mi', 'largest_city_wl', 'area_water_sq_mi', 'named_for']
#classes = ['district']

models = []
for class_ in classes:
    print("CLASS: {}\n".format(class_))
    cnn_blstm = CNN_BLSTM(class_, EPOCHS, DROPOUT,
                          DROPOUT_RECURRENT, LSTM_STATE_SIZE,
                          CONV_SIZE, LEARNING_RATE, OPTIMIZER)
    cnn_blstm.loadData(class_)
    cnn_blstm.addCharInfo()
    cnn_blstm.embed()
    cnn_blstm.createBatches()
    cnn_blstm.buildModel()
    cnn_blstm.train()
    models.append([class_, cnn_blstm])
    print("\n")

print('ALL TRAINED MODELS: {}'.format(len(models)))
