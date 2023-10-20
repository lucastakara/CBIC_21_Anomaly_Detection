import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from main import plot_anomaly_points, plot_train_val_loss, determine_anomalies


class TestPlotFunctions(unittest.TestCase):

    @patch('main.plt.show')  # Mock 'plt.show' to avoid opening a window during testing
    def test_plot_anomaly_points(self, mock_plt):
        # Mock data - adjust as necessary to match what your function expects and uses
        anomalies = pd.DataFrame({
            'Date': pd.to_datetime(['2021-01-01', '2021-01-02']),
            'Close': [100, 150]
        })
        test_score_df = pd.DataFrame({
            'Date': pd.to_datetime(['2021-01-01', '2021-01-02']),
            'Close': [100, 150]
        })
        train = pd.DataFrame({
            'Date': pd.to_datetime(['2020-12-30', '2020-12-31']),
            'Close': [80, 90]
        })
        test = pd.DataFrame({
            'Date': pd.to_datetime(['2021-01-01', '2021-01-02']),
            'Close': [100, 150]
        })

        # Call the function to be tested
        plot_anomaly_points(anomalies, test_score_df, train, test)

        # If the function executes plot commands, the plt.show() method should be called
        mock_plt.assert_called_once()

    @patch('main.plt.show')
    def test_plot_train_val_loss(self, mock_plt):
        # Mock data for history object - adjust as necessary
        history_mock = Mock()
        history_mock.history = {'loss': [0.1, 0.2], 'val_loss': [0.15, 0.25]}

        # Call the function to be tested
        plot_train_val_loss(history_mock)

        # Check if plt.show() was called
        mock_plt.assert_called_once()


class TestDetermineAnomaliesFunction(unittest.TestCase):
    @patch('main.plt.show')  # Assuming the function is in 'main.py'
    def test_determine_anomalies(self, mock_plt):
        # Mocking the model's predict method
        model_mock = Mock()
        model_mock.predict.side_effect = lambda x: x + 0.1  # For example, the prediction is the input + some noise

        TIME_STEPS = 30  # This should match the setting in your actual function

        # Generating data for X_train and X_test with an understanding that each sequence is based on TIME_STEPS
        num_sequences_train = 100  # arbitrary choice
        num_sequences_test = 20  # arbitrary choice, but should reflect that it's less than the train set
        X_train = np.random.rand(num_sequences_train, TIME_STEPS, 1)  # (num_sequences, TIME_STEPS, num_features)
        X_test = np.random.rand(num_sequences_test, TIME_STEPS, 1)

        # Remember, we will "lose" TIME_STEPS - 1 points due to the sliding window
        test_dates = pd.date_range(start='2021-01-01', periods=num_sequences_test + TIME_STEPS - 1, freq='D')
        test_close_prices = np.random.uniform(low=100, high=200, size=(num_sequences_test + TIME_STEPS - 1,))
        test_df = pd.DataFrame({'Date': test_dates, 'Close': test_close_prices})

        # Calling the function with the mock data and model
        result_test_score_df, result_anomalies = determine_anomalies(model_mock, X_train, X_test, test_df, TIME_STEPS)

        expected_length = num_sequences_test  # since we lose TIME_STEPS - 1 points, and we generated that many extra
        self.assertEqual(len(result_test_score_df), expected_length)

        # Ensure that the plotting function was called
        mock_plt.assert_called()


if __name__ == '__main__':
    unittest.main()
