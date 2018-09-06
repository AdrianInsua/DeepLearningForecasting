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

from config import GROUP_BY

class Prediction:
    """ Factory """
    def __init__(self, data, mode, field, v):
        self.data = data
        self.train_data, self.test_data = data, data
        self.verbose = v
        self.pre = Preprocess(data, field, v)
        self.mode = mode
        self.algorithm = None

    def preprocess(self, group, decompose, scale, supervised, split):
        """Preprocess data"""

        if group:
            self.pre.group_by(GROUP_BY)
        if decompose:
            self.pre.decompose_seasonality()
        if scale:
            self.pre.scale_data()
        if supervised:
            self.pre.timeseries_to_supervised()
        if split:
            self.train_data, self.test_data = self.pre.split_data()
    
    def init_model(self):
        """Initialize model"""
        print(self.mode)
        if self.mode == 'DEEP':
            self.algorithm = TimeSeriesPrediction(self.train_data, self.test_data, self.verbose)
        elif self.mode == 'ML':
            self.algorithm = TSM(self.train_data, self.test_data, self.verbose)

    def train(self):
        """Model training"""

        self.algorithm.train()

    def predict(self):
        """Predict data"""

        self.algorithm.predict()

    def load_pretrained(self):
        """Load a pretrained model to do predictions"""
        print('Need implementation')
