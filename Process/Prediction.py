"""
    Factory class to select between multiple training algorithms
    mode = 'LSTM' -> DeepLearning LSTM
    else -> SVR
"""
# Preprocess Class
from Aux.Preprocess import Preprocess

# Deep learning Prediction
from Process.DeepLearning.TimeSeriesPrediction import TimeSeriesPrediction

# machine learning
from Process.MachineLearning.TimeSeriesPrediction import TimeSeriesPrediction as TSM

# plot
import matplotlib.pyplot as plt

import numpy as np

from config import GROUP_BY, STEPS

class Prediction:
    """ Factory """
    def __init__(self, data, mode, field, v):
        self.data = data
        self.pred_data = data[-14:]
        self.train_data, self.test_data = data, data
        self.verbose = v
        self.pre = Preprocess(data, field, v)
        self.mode = mode
        self.scaler = None
        self.algorithm = None

    def preprocess(self, group, decompose, scale, supervised, split):
        """Preprocess data"""

        if group:
            self.pre.group_by(GROUP_BY)
        if decompose:
            self.pre.decompose_seasonality()
        if scale:
            self.scaler, scaled = self.pre.scale_data()
        if supervised:
            self.pre.timeseries_to_supervised()
        if split:
            self.train_data, self.test_data = self.pre.split_data()

        self.pred_data = scaled[-12:]

    def preprocess_seasonality(self, field, scale, supervised, split):
        """Preprocess seasonality data"""

        self.pre.change_field(field)
        if scale:
            self.pre.scale_data()
        if supervised:
            self.pre.timeseries_to_supervised()
        if split:
            self.train_data, self.test_data = self.pre.split_data()

    def init_model(self):
        """Initialize model"""

        if self.mode == 'DEEP':
            self.algorithm = TimeSeriesPrediction(self.train_data, self.test_data, self.verbose)
        elif self.mode == 'ML':
            self.algorithm = TSM(self.train_data, self.test_data, self.verbose)

    def train(self):
        """Model training"""

        self.algorithm.train()

    def evaluate(self):
        """Evaluate on test data"""

        pred = self.algorithm.predict(self.test_data['x'], self.test_data['y'], self.scaler)

        return self.pre.rescale_data(pred)

    def predict(self):
        """Predict data"""

        ts_data = self.pre.pred_to_supervised(self.pred_data)
        print(ts_data)

        pred = {
            'x': np.array(ts_data[['x' + str(x) for x in range(0, STEPS)]].values),
            'y': np.array(ts_data['y'].values)
        }

        print(pred)

        pred = self.algorithm.predict(pred['x'], pred['y'], self.scaler)

        return self.pre.rescale_data(pred)

    def show_pred_seasonality(self, pred):
        plt.figure()
        plt.plot(pred, label='pred')
        plt.plot(self.test_data['y'], label='test')
        plt.show()

    def load_pretrained(self):
        """Load a pretrained model to do predictions"""
        print('Need implementation')
