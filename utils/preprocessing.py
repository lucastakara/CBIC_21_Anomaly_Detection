import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import numpy as np

class Dataset:
    def __init__(self):
        """Initializes the Dataset object by loading the data."""
        self._data = self._load_data()

    @property
    def data(self):
        """
        A property (getter) that provides access to the loaded data.

        :return: A DataFrame containing the loaded data.
        """
        return self._data

    @staticmethod
    def _load_data():
        """
        Loads stock data from Yahoo Finance.

        :return: A DataFrame containing the stock data.
        """
        yf.pdr_override()
        raw_data = pdr.get_data_yahoo("^BVSP", start='1993-04-28', end="2021-05-18")
        processed_data = Dataset.process_raw_data(raw_data)
        return processed_data

    @staticmethod
    def process_raw_data(raw_data):
        """
        Processes raw stock data for analysis.

        :param raw_data: DataFrame containing raw stock data.
        :return: A DataFrame with processed stock data.
        """
        processed_data = raw_data.copy()
        processed_data.reset_index(inplace=True, drop=False)
        processed_data = processed_data[['Date', 'Close']]
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        return processed_data

    @staticmethod
    def split_data(df, train_size=0.8):
        """
        Splits the DataFrame into training and testing datasets.

        :param df: DataFrame to split.
        :param train_size: float, proportion of the dataset to include in the train split (0 < train_size < 1).
        :return: Two DataFrames representing the training and testing datasets.
        """
        if not (0 < train_size < 1):
            raise ValueError("train_size must be in the range (0, 1)")

        split_point = int(len(df) * train_size)
        split_date = df.iloc[split_point]['Date']

        train = df[df['Date'] < split_date]
        test = df[df['Date'] >= split_date]

        return train, test

    def prepare_data_for_autoencoder(self, train, test, sequence_length=30):
        """
        Prepares and scales data for autoencoder training.

        :param train: DataFrame containing the training data.
        :param test: DataFrame containing the testing data.
        :param sequence_length: int, length of the sequence for the autoencoder.
        :return: Scaled training and testing data, along with the scaler object.
        """
        train_scaled, test_scaled = self.scale_data(train, test)
        X_train, y_train = self.create_sequences(train_scaled, sequence_length)
        X_test, y_test = self.create_sequences(test_scaled, sequence_length)

        print(X_train.shape)
        print(X_test.shape)
        return X_train, y_train, X_test, y_test

    def scale_data(self, train, test):
        """
        Scales 'Close' price data in training and testing datasets.

        :param train: DataFrame containing the training data.
        :param test: DataFrame containing the testing data.
        :return: DataFrames of scaled 'Close' prices for training and testing data, and the scaler object.
        """
        scaler = StandardScaler()
        train_scaled = train.copy()
        test_scaled = test.copy()

        scaler.fit(train_scaled[['Close']])
        train_scaled['Close'] = scaler.transform(train_scaled[['Close']])
        test_scaled['Close'] = scaler.transform(test_scaled[['Close']])

        return train_scaled, test_scaled

    @staticmethod
    def create_sequences(data, sequence_length):
        """
        Creates sequences of data for time series forecasting with an autoencoder.

        :param data: DataFrame containing 'Close' price data.
        :param sequence_length: int, the number of time steps for each sequence.
        :return: Numpy arrays X and y containing the sequences and their corresponding labels.
        """
        X, y = [], []

        # Adjusting the range to accommodate sequences for all data points
        for i in range(len(data) - sequence_length + 1):
            # Extract sequences for the 'Close' prices
            sequence = data['Close'].iloc[i:(i + sequence_length)].values
            label = data['Close'].iloc[i + sequence_length - 1]

            # Append the sequences
            X.append(sequence)
            y.append(label)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Reshape X to three dimensions (num_samples, sequence_length, num_features)
        # Currently, num_features is 1 as you're only using 'Close' prices for predictions.
        X = np.expand_dims(X, axis=-1)  # Adding a new dimension for num_features
        y = np.expand_dims(y, axis=-1)  # Adding a new dimension for num_features
        return X, y
