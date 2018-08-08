import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, concat
from sklearn import preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

class TimeSeriesSVR:
    def __init__(self, data):
        data[0] = data[0].interpolate(method= 'quadratic')
        print(data[0])
        new_train = self.series_to_supervised(data[0].values, 1, 1)
        self.X_train = new_train[['var1(t-1)']]
        print(self.X_train)
        self.Y_train = new_train['var1(t)']
        new_test = self.series_to_supervised(data[1].values, 1, 1)
        self.X_test = new_test[['var1(t-1)']]
        self.Y_test = new_test['var1(t)']

        print(self.X_train)
        scaler = pre.StandardScaler().fit(self.X_train)
        self.X_train_scaled = scaler.transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        self.SVR_model = SVR(kernel='rbf',C=100,gamma=.001)

    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
    	"""
    	Frame a time series as a supervised learning dataset.
    	Arguments:
    		data: Sequence of observations as a list or NumPy array.
    		n_in: Number of lag observations as input (X).
    		n_out: Number of observations as output (y).
    		dropnan: Boolean whether or not to drop rows with NaN values.
    	Returns:
    		Pandas DataFrame of series framed for supervised learning.
    	"""
    	n_vars = 1 if type(data) is list else data.shape[1]
    	df = DataFrame(data)
    	cols, names = list(), list()
    	# input sequence (t-n, ... t-1)
    	for i in range(n_in, 0, -1):
    		cols.append(df.shift(i))
    		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    	# forecast sequence (t, t+1, ... t+n)
    	for i in range(0, n_out):
    		cols.append(df.shift(-i))
    		if i == 0:
    			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    		else:
    			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	# put it all together
    	agg = concat(cols, axis=1)
    	agg.columns = names
    	# drop rows with NaN values
    	if dropnan:
    		agg.dropna(inplace=True)
    	return agg

    def train(self):
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100], 'gamma':[0.001, 0.05, 1]}
        # self.SVR_model = SVR(kernel='rbf',C=100,gamma=.001).fit(self.X_train_scaled,self.Y_train)
        svr = SVR()
        clf = GridSearchCV(svr, parameters, verbose=1)
        clf.fit(self.X_train_scaled, self.Y_train)
        print(clf.cv_results_)

    def predict(self):
        predictions = self.SVR_model.predict(self.X_test)
        print(predictions)
        with plt.style.context("fivethirtyeight"):
            plt.plot(self.Y_test, label='Real')
            plt.plot(predictions, label='Pred')
            plt.show()
