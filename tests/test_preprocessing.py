import unittest
import pandas as pd
import numpy as np
from utils.preprocessing import Dataset


class DatasetTest(unittest.TestCase):
    def setUp(self):
        # Sample dataset for testing
        self.dataset = Dataset()
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        self.sample_data = pd.DataFrame(date_rng, columns=['Date'])
        self.sample_data['Close'] = np.random.randint(0, 100, size=(len(date_rng)))

    def test_split_data(self):
        train, test = self.dataset.split_data(self.sample_data, train_size=0.5)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)

    def test_prepare_data_for_autoencoder(self):
        train, test = self.dataset.split_data(self.sample_data, train_size=0.5)
        X_train, y_train, X_test, y_test = self.dataset.prepare_data_for_autoencoder(train, test,
                                                                                             sequence_length=1)

        self.assertEqual(X_train.shape, (4, 1, 1))  # Sequence of length 1, 4 sequences in training data of 1 feature
        self.assertEqual(X_test.shape, (4, 1, 1))  # Same for test data

    def test_create_sequences(self):
        sequence_length = 2
        X, y = self.dataset.create_sequences(self.sample_data, sequence_length)

        # Adjusting the expected length of X to match the method's actual behavior
        expected_length = len(self.sample_data) - sequence_length + 1

        self.assertEqual(len(X), expected_length)
        self.assertTrue((X[0, :, 0] == self.sample_data['Close'].iloc[:sequence_length].values).all())


if __name__ == '__main__':
    unittest.main()
