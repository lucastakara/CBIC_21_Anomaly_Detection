import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class Dataset:

    def __init__(self):
        self.get_data = self.load_df()

    @staticmethod
    def load_df():
        df = web.DataReader("^BVSP", data_source="yahoo", start='04/28/1993', end="05/18/2021")
        # Data from index to column
        df.reset_index(inplace=True)
        # Selecting only Date and Close prices
        df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    @staticmethod
    def train_test_split(df):
        # Train 0.7, Test 0.2 , Val 0.1
        train, test = df.loc[df['Date'] <= '2015-09-30'], df.loc[df['Date'] > '2015-09-30']
        return train, test

    def preprocess_df(self, train, test):
        # Scale the data for LSTM
        scaler = StandardScaler()
        scaler = scaler.fit(train[['Close']])
        train['Close'] = scaler.transform(train[['Close']])
        test['Close'] = scaler.transform(test[['Close']])
        # Creating sequences to feed LSTM
        X_train, y_train = create_sequences(train[['Close']], train['Close'], 30)
        X_test, y_test = create_sequences(test[['Close']], test['Close'], 30)
        print(f'Training shape: {X_train.shape}')
        print(f'Testing shape: {X_test.shape}')

        return X_train, y_train, X_test, y_test


def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)

