import unittest
import numpy as np
from models_architecture.LSTM_Autoencoder import LSTM_Autoencoder  # Replace 'your_module' with the actual module name
import tensorflow as tf


class TestLSTMAutoencoder(unittest.TestCase):
    def setUp(self):
        self.time_window_size = 30
        self.n_features = 1
        self.input_shape = (self.time_window_size, self.n_features)
        self.autoencoder = LSTM_Autoencoder(self.input_shape)

        # Creating dummy data for training and testing
        self.X_train = np.random.rand(100, self.time_window_size, self.n_features)
        self.y_train = np.random.rand(100, self.time_window_size, self.n_features)
        self.X_test = np.random.rand(20, self.time_window_size, self.n_features)
        self.y_test = np.random.rand(20, self.time_window_size, self.n_features)

    def test_model_creation(self):
        self.assertIsInstance(self.autoencoder.model, tf.keras.Model)
        self.assertEqual(self.autoencoder.model.input_shape, (None, *self.input_shape))

    def test_training(self):
        history = self.autoencoder.train(self.X_train, self.y_train, epochs=1)  # Using 1 epoch for quick testing
        self.assertIn('loss', history.history)

    def test_prediction(self):
        predictions = self.autoencoder.predict(self.X_train)
        self.assertEqual(predictions.shape, self.X_train.shape)

    def test_evaluation(self):
        loss = self.autoencoder.evaluate(self.X_test, self.y_test)
        self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main()
