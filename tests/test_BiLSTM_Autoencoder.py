import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from models_architectures.BILSTM_Autoencoder import BiLSTMAutoencoder  # Replace with your actual import statement


class TestBiLSTMAutoencoder(unittest.TestCase):

    def setUp(self):
        # This method will be used to set up values that will be used in multiple tests.
        self.input_shape = (30, 10)  # Example input shape, adjust as necessary
        self.autoencoder = BiLSTMAutoencoder(self.input_shape)

    def test_model_creation(self):
        """
        Test the successful creation of a BiLSTM Autoencoder model.
        """
        self.assertIsNotNone(self.autoencoder.model)

    @patch('tensorflow.keras.models.Sequential.fit')
    def test_model_training(self, mock_fit):
        """
        Test the training process of the model.
        """
        # Create dummy data
        X_train = np.random.rand(100, *self.input_shape)
        y_train = np.random.rand(100, self.input_shape[0])  # Adjust based on your actual output shape

        # Assume these parameters for the test
        epochs = 10
        batch_size = 32
        validation_split = 0.1
        shuffle = True

        # Mock the behavior of the fit method and return a dummy history object
        mock_fit.return_value = MagicMock(history={'loss': [0.1, 0.05]})

        # Train the model
        history = self.autoencoder.train(X_train, y_train, epochs, batch_size, validation_split, shuffle)

        # Check that the fit method was called with the correct arguments
        mock_fit.assert_called_with(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)

        # Check that the history object contains loss information
        self.assertIn('loss', history.history)

    @patch('tensorflow.keras.models.Sequential.predict')
    def test_model_prediction(self, mock_predict):
        """
        Test the prediction process of the model.
        """
        # Create dummy data for prediction
        data = np.random.rand(5, *self.input_shape)

        # Mock the behavior of the predict method
        mock_prediction = np.random.rand(5, self.input_shape[0])  # Adjust based on your actual output shape
        mock_predict.return_value = mock_prediction

        # Make a prediction
        prediction = self.autoencoder.predict(data)

        # Check that the predict method was called with the correct argument
        mock_predict.assert_called_with(data)

        # Check that the prediction returned is as expected
        np.testing.assert_array_equal(prediction, mock_prediction)

    @patch('tensorflow.keras.models.Sequential.evaluate')
    def test_model_evaluation(self, mock_evaluate):
        """
        Test the evaluation process of the model.
        """
        # Create dummy test data and labels
        X_test = np.random.rand(20, *self.input_shape)
        y_test = np.random.rand(20, self.input_shape[0])  # Adjust based on your actual output shape

        # Mock the behavior of the evaluate method
        mock_loss = 0.05  # A dummy loss value
        mock_evaluate.return_value = mock_loss

        # Evaluate the model
        loss = self.autoencoder.evaluate(X_test, y_test)

        # Check that the evaluate method was called with the correct arguments
        mock_evaluate.assert_called_with(X_test, y_test)

        # Check that the loss is as expected
        self.assertEqual(loss, mock_loss)


if __name__ == '__main__':
    unittest.main()
