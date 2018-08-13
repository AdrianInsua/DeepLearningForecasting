"""
    Deep Learning LSTM Time series predictions
"""

# Deep learning libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras import callbacks
import numpy as np
from config import BATCH_SIZE, NEURONS, NB_EPOCH, LR, STEPS
import matplotlib.pyplot as plt

np.random.seed(7)

class TimeSeriesPrediction:
    """Time series predicion class"""
    def __init__(self, train, test, v):
        self.verbose = v
        self.model = None
        self.x_train, self.y_train = train.x, np.asarray(train.y)
        self.x_test, self.y_test = test.x, np.asarray(test.y)
        self.x_train = np.reshape(self.x_train.values, (self.x_train.shape[0], 1, 2))
        print(self.x_train)
        self.x_test = np.reshape(self.x_test.values, (self.x_test.shape[0], 1, 1))

        self.create_model()

    def create_model(self):
        """Create prediction model"""

        self.model = Sequential()
        self.model.add(LSTM(NEURONS, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(Dense(1))
        adam = optimizers.adam(lr=LR)
        self.model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])

    def predict(self):
        """walk-forward validation on the test data"""

        yhat = self.model.predict(self.x_test)

        if self.verbose >= 1:
            plt.plot(yhat, label='predict')
            plt.plot(self.y_test, label='true')
            plt.legend()
            plt.show()

    def train(self):
        """Fit a LSTM network to train data"""

        if self.verbose >= 1:
            print('Training model...')

        cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=24)]
        hist = self.model.fit(self.x_train, self.y_train, epochs=NB_EPOCH,\
         callbacks=cbks, shuffle=False, batch_size=BATCH_SIZE, \
        #  validation_data=(self.x_test, self.y_test),\
        validation_split=.15,\
        verbose=1 if self.verbose >= 1 else 0)
        
        if self.verbose >= 2:
            plt.figure(1)
            plt.subplot(211)
            plt.plot(hist.history['acc'], label='acc')
            plt.plot(hist.history['val_acc'], label='val_acc')
            plt.legend()
            plt.title('Accuracy')
            plt.subplot(212)
            plt.plot(hist.history['val_loss'], label='acc')
            plt.legend()
            plt.title('Val loss')
            plt.show()
