"""
    Deep Learning LSTM Time series predictions
"""

# Deep learning libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import ConvLSTM2D
from keras import optimizers
from keras import callbacks

# Common libraries
import numpy as np
import matplotlib.pyplot as plt

# Global variables
from config import BATCH_SIZE, NEURONS, NB_EPOCH, LR, STEPS, LAYERS, MODE, MODEL

np.random.seed(7)

class TimeSeriesPrediction:
    """Time series predicion class"""
    def __init__(self, train, test, v):
        self.verbose = v
        self.model = None
        self.x_train, self.y_train = train['x'], train['y']
        self.x_test, self.y_test = test['x'], test['y']
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))

        self.create_model()

    def create_model(self):
        """Create prediction model"""

        self.model = Sequential()
        if MODEL == 'LSTM':
            for xl in range(LAYERS):
                if xl == 0:
                    self.model.add(LSTM(NEURONS, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), return_sequences=LAYERS > 1))
                if LAYERS > 1:
                    self.model.add(LSTM(NEURONS, return_sequences=xl < LAYERS - 1))
                if xl < LAYERS - 1:
                    self.model.add(Dropout(0.2))
        elif MODEL == 'CNN':
            self.model.add(ConvLSTM2D(5, 5))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64))
        self.model.add(Dropout(128))
        self.model.add(Dense(1))
        adam = optimizers.adam(lr=LR)
        self.model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    def predict(self):
        """walk-forward validation on the test data"""

        yhat = self.model.predict(self.x_test)

        if self.verbose >= 2:
            plt.plot(yhat, label='predict')
            plt.plot(self.y_test, label='true')
            plt.legend()
            plt.show()

        return yhat

    def train(self):
        """Fit a LSTM network to train data"""

        if self.verbose >= 1:
            print('Training model...')

        cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=24)]
        hist = self.model.fit(self.x_train, self.y_train, epochs=NB_EPOCH,\
         callbacks=cbks, shuffle=False, batch_size=BATCH_SIZE, \
        #  validation_data=(self.x_test, self.y_test),\
        validation_data=(self.x_test, self.y_test),\
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
