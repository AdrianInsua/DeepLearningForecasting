"""
    Preprocess data
        *Scale
        *Transform in supervised data
        *Split
"""

#imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

from config import SPLIT, STEPS

class Preprocess:
    """Main class"""
    def __init__(self, data, field, v):
        self.data = data
        self.field = field
        self.scaled = data[field]
        self.scaler = {}
        self.verbose = v

    def group_by(self, group='M'):
        """grouping method"""

        if self.verbose >= 1:
            print('Grouping data by ' + group + '...')

        self.data[self.field].replace(0, np.nan, inplace=True)
        self.data[self.field].fillna(method='ffill', inplace=True)
        self.data[self.field].fillna(method='bfill', inplace=True)
        self.data.reset_index(inplace=True)
        self.data['ID_FECHA'] = pd.to_datetime(self.data['ID_FECHA'])
        self.data = self.data.set_index('ID_FECHA')
        self.data = self.data.groupby(pd.TimeGrouper(freq=group)).sum()

    def decompose_seasonality(self):
        """decomponse data seasonality"""

        if self.verbose >= 1:
            print('Decomposing seasonality...')

        result = seasonal_decompose(self.data[self.field], freq=7)

        self.data['seasonal'] = result.seasonal
        self.data['seasonal'].fillna(method='ffill', inplace=True)
        self.data['seasonal'].fillna(method='bfill', inplace=True)
        self.data['trend'] = result.trend
        self.data['trend'].fillna(method='ffill', inplace=True)
        self.data['trend'].fillna(method='bfill', inplace=True)
        self.data['resid'] = result.resid
        self.data['resid'].fillna(method='ffill', inplace=True)
        self.data['resid'].fillna(method='bfill', inplace=True)

        if self.verbose >= 2:
            result.plot()
            plt.show()

    def scale_data(self):
        """transform data to be stationary"""

        if self.verbose >= 1:
            print("Scaling data...")

        values = self.data[self.field].values.reshape(-1,1)
        values = values.astype('float64')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled = self.scaler.fit_transform(values)
        self.data = self.scaled

    def timeseries_to_supervised(self):
        """Convert timeseries to supervised"""

        if self.verbose >= 1:
            print("transforming data to supervised data...")

        columns = pd.DataFrame({'x': self.data[:, 0]}).shift(STEPS)
        columns['y'] = self.data[:,0]
        self.data = columns
        self.data.fillna(0, inplace=True)

        if self.verbose >= 2:
            print('Timeseries data: \n', self.data)

    def split_data(self):
        """Split data into train and test"""

        if self.verbose >= 1:
            print("splitting data...")

        split = round(len(self.data) * SPLIT)

        train, test = self.data[['x', 'y']][:split], self.data[['x', 'y']][split:]
        return train, test

def series_to_supervised(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
