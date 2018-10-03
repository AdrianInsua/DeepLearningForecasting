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
    def __init__(self, v=1):
        self.scaler = {}
        self.verbose = v

    def change_field(self, field):
        """reasing field value"""

        self.field = field

    def group_by(self, data, field, group='W'):
        """grouping method"""

        if self.verbose >= 1:
            print('Grouping data by ' + str(group) + '...')

        data[field].replace(0, np.nan, inplace=True)
        data[field].fillna(method='ffill', inplace=True)
        data[field].fillna(method='bfill', inplace=True)
        data.reset_index(inplace=True)
        data['ID_FECHA'] = pd.to_datetime(data['ID_FECHA'])
        data = data.set_index('ID_FECHA', drop=True)
        data = data.groupby(pd.TimeGrouper(freq=group)).sum()

        if self.verbose >= 2:
            plot_data(data, 'grouped data')

        return data

    def decompose_seasonality(self, data, field):
        """decomponse data seasonality"""

        if self.verbose >= 1:
            print('Decomposing seasonality...')

        result = seasonal_decompose(data[field], freq=7)

        data['seasonal'] = result.seasonal
        data['seasonal'].fillna(method='ffill', inplace=True)
        data['seasonal'].fillna(method='bfill', inplace=True)
        data['trend'] = result.trend
        data['trend'].fillna(method='ffill', inplace=True)
        data['trend'].fillna(method='bfill', inplace=True)
        data['resid'] = result.resid
        data['resid'].fillna(method='ffill', inplace=True)
        data['resid'].fillna(method='bfill', inplace=True)

        if self.verbose >= 2:
            result.plot()
            plt.show()

        return data

    def scale_data(self, data, field):
        """transform data to be stationary"""

        if self.verbose >= 1:
            print("Scaling data...")

        values = data[field].values.reshape(-1,1)
        values = values.astype('float64')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)

        if self.verbose >= 2:
            plot_data(scaled, 'scaled data', 'scaled data')

        return self.scaler, scaled

    def rescale_data(self, pred):
        """transform data to original range"""

        if self.verbose >= 1:
            print("Rescaling data...")

        return self.scaler.inverse_transform(pred.reshape(-1, 1))

    def transform_to_scale(self, data, field):
        """transform new data to current scale"""

        if self.verbose >= 1:
            print("Scaling new data...")

        values = data[field].values.reshape(-1,1)
        values = values.astype('float64')

        return self.scaler.transform(values)

    def timeseries_to_supervised(self, data):
        """Convert timeseries to supervised"""

        if self.verbose >= 1:
            print("transforming data to supervised data...")

        columns = pd.DataFrame()

        for xr in reversed(range(1, STEPS + 1)):
            aux = pd.DataFrame({'x': data[:, 0]}).shift(xr)
            columns['x'+str(xr)] = aux.x
        columns.fillna(0, inplace=True)
        columns['y'] = data[:,0]
        data = columns[:-STEPS + 1]
        data.fillna(0, inplace=True)

        if self.verbose >= 2:
            print('Timeseries data: \n', data)
            plot_data(data, 'timeseries data')

        return data

    def pred_to_supervised(self, data):
        """Convert prediction data to supervised"""

        if self.verbose >= 1:
            print("transforming data to supervised data...")

        # pdb.set_trace()
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

    def split_data(self, data):
        """Split data into train and test"""

        if self.verbose >= 1:
            print("splitting data...")

        split = round(len(data[:-12]) * SPLIT)
        train = {
            'x': np.array(data[['x' + str(x) for x in reversed(range(1, STEPS + 1))]].values)[:split],
            'y': np.array(data['y'].values)[:split]
        }
        test = {
            'x': np.array(data[['x' + str(x) for x in reversed(range(1, STEPS + 1))]].values)[split:],
            'y': np.array(data['y'].values)[split:]
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