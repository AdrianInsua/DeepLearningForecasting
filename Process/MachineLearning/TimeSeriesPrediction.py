"""
    Machine Learning LSTM Time series predictions
"""

# Machine learning libraries
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# common libraries
import numpy as np
import matplotlib.pyplot as plt

# global variables
from config import MODEL

parameters = {'C': [1,10,100]}

np.random.seed(0)

class TimeSeriesPrediction:
    def __init__(self, train, test, v):
        self.verbose = v
        self.model = None
        self.x_train, self.y_train = train['x'], train['y']
        self.x_test, self.y_test = test['x'], test['y']

        self.create_model()

    def create_model(self):
        """Create prediction model"""

        if self.verbose >= 1:
            print('Initializing model...')

        if MODEL == 'SVR':
            self.model = SVR(C=1, kernel='linear')

    def train(self):
        """Fit model"""

        if self.verbose >= 1:
            print('Training model...')

        grid = GridSearchCV(self.model, cv=3, n_jobs=1, param_grid=parameters)
        grid.fit(self.x_train, self.y_train)
        print(np.array(grid.cv_results_['mean_test_score']))
        print(grid.best_params_)
        hist = self.model.fit(self.x_train, self.y_train)

        if self.verbose >= 2:
            print(hist)

    def predict(self):
        """Predict test"""

        if self.verbose >= 1:
            print('Predicting data...')

        pred = self.model.predict(self.x_test)

        if self.verbose >= 1:
            plt.plot(pred, label='predict')
            plt.plot(self.y_test, label='true')
            plt.legend()
            plt.show()
        
        return pred
