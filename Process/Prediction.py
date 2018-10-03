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

from config import GROUP_BY, STEPS, SEASON_FIELD

class Prediction:
    """ Factory """
    def __init__(self, mode, field, group_by, v):
        self.field = field
        self.group_by = group_by
        self.verbose = v
        self.pre = Preprocess(v)
        self.mode = mode

    def preprocess(self, data, group, decompose, scale, supervised, split):
        """Preprocess data"""

        train_data, test_data = None, None
        if group:
            data = self.pre.group_by(data, self.field, self.group_by)
        if decompose:
            data = self.pre.decompose_seasonality(data, self.field)
        if scale:
            scaler, data = self.pre.scale_data(data, self.field)
        if supervised:
            data = self.pre.timeseries_to_supervised(data)
        if split:
            train_data, test_data = self.pre.split_data(data[:-12])

        pred_data = data[-12:]

        return data, scaler, train_data, test_data, pred_data

    def preprocess_seasonality(self, data, field, decompose, scale, supervised, split):
        """Preprocess seasonality data"""
        dec_data, scaler, train_data, test_data, pred_data = None, None, None, None, None
        if decompose:
            data = self.pre.group_by(data, self.field, self.group_by)
            data = self.pre.decompose_seasonality(data, self.field)
        if scale:
            scaler, dec_data = self.pre.scale_data(data, field)
        if supervised:
            dec_data = self.pre.timeseries_to_supervised(dec_data)
        if split:
            train_data, test_data = self.pre.split_data(dec_data[:-12])

        pred_data = dec_data[-12:] if dec_data is not None else None

        return data, dec_data, scaler, train_data, test_data, pred_data

    def init_model(self, data_shape):
        """Initialize model"""

        if self.mode == 'DEEP':
            self.algorithm = TimeSeriesPrediction(data_shape, self.verbose)
        elif self.mode == 'ML':
            self.algorithm = TSM(self.verbose)

    def train(self, train_data, test_data):
        """Model training"""

        self.algorithm.train(train_data, test_data)

    def evaluate(self, test_data, scaler):
        """Evaluate on test data"""

        pred = self.algorithm.predict(test_data['x'], test_data['y'], scaler)

        return self.pre.rescale_data(pred)

    def predict(self, pred_data, scaler):
        """Predict data"""

        ts_data = pred_data
        print(ts_data)

        pred = {
            'x': np.array(ts_data[['x' + str(x) for x in reversed(range(1, STEPS + 1))]].values),
            'y': np.array(ts_data['y'].values)
        }

        print(pred)

        pred = self.algorithm.predict(pred['x'], pred['y'], scaler)

        return self.pre.rescale_data(pred)

    def show_pred_seasonality(self, pred):
        plt.figure()
        plt.plot(pred, label='pred')
        plt.plot(self.test_data['y'], label='test')
        plt.show()

    def load_pretrained(self):
        """Load a pretrained model to do predictions"""
        print('Need implementation')
