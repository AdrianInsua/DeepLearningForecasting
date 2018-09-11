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
import pdb

from config import SPLIT, STEPS

class Preprocess:
    """Main class"""
    def __init__(self, data, field, v):
        self.data = data
        self.field = field
        self.scaled = data[field]
        self.scaler = {}
        self.verbose = v

    def change_field(self, field):
        """reasing field value"""

        self.field = field

    def group_by(self, group='W'):
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

        if self.verbose >= 2:
            plot_data(self.data, 'grouped data')

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
            plot_data(self.data, 'seasonality data')
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

        if self.verbose >= 2:
            plot_data(self.data, 'scaled data', 'scaled data')

        return self.scaler, self.scaled

    def rescale_data(self, pred):
        """transform data to original range"""

        if self.verbose >= 1:
            print("Rescaling data...")

        return self.scaler.inverse_transform(pred.reshape(-1, 1))

    def transform_to_scale(self, data):
        """transform new data to current scale"""

        if self.verbose >= 1:
            print("Scaling new data...")

        values = data[self.field].values.reshape(-1,1)
        values = values.astype('float64')

        return self.scaler.transform(values)

    def timeseries_to_supervised(self):
        """Convert timeseries to supervised"""

        if self.verbose >= 1:
            print("transforming data to supervised data...")

        columns = pd.DataFrame()
        for xr in reversed(range(1, STEPS + 1)):
            aux = pd.DataFrame({'x': self.scaled[:, 0]}).shift(xr)
            columns['x'+str(xr)] = aux.x
        columns.fillna(0, inplace=True)
        columns['y'] = self.scaled[:,0]
        self.scaled = columns[:-STEPS + 1]
        self.scaled.fillna(0, inplace=True)

        if self.verbose >= 2:
            print('Timeseries data: \n', self.scaled)
            plot_data(self.scaled, 'timeseries data')

    def pred_to_supervised(self, data):
        """Convert prediction data to supervised"""

        if self.verbose >= 1:
            print("transforming data to supervised data...")

        columns = pd.DataFrame()
        for xr in range(0, STEPS + 1):
            name = 'x'+str(xr) if xr < STEPS else 'y'
            aux = pd.DataFrame({str(name): data[:, 0]}).shift(-xr)
            columns[str(name)] = aux[str(name)]

        columns.fillna(0, inplace=True)
        data = columns[:-STEPS + 1]
        data.fillna(0, inplace=True)

        if self.verbose >= 2:
            print('Timeseries data: \n', data)
            plot_data(data, 'timeseries data')

        return data

    def split_data(self):
        """Split data into train and test"""

        if self.verbose >= 1:
            print("splitting data...")

        split = round(len(self.scaled[:-12]) * SPLIT)
        train = {
            'x': np.array(self.scaled[['x' + str(x) for x in reversed(range(1, STEPS + 1))]].values)[:split],
            'y': np.array(self.scaled['y'].values)[:split]
        }
        test = {
            'x': np.array(self.scaled[['x' + str(x) for x in reversed(range(1, STEPS + 1))]].values)[split:],
            'y': np.array(self.scaled['y'].values)[split:]
        }
        
        if self.verbose >= 2:
            plot_data(train['x'], 'train split')
            plot_data(test['x'], 'test split')
        return train, test

def series_to_supervised(dataset, look_back):
    """transform timeseries into supervised"""
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


def plot_data(data, title, label=''):
    """plot data"""
    plt.figure()
    plt.title(title)
    if data is pd:
        data.plot()
    else:
        plt.plot(data, label=label)
    plt.legend()
    plt.show()