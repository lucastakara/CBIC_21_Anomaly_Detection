from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional


class BiLSTMAutoencoder:
    def __init__(self, input_shape):
        """
        Initializes the BiLSTM Autoencoder model.

        :param input_shape: tuple, shape (time_window_size, n_features) for LSTM input.
        """
        self.model = self._create_model(input_shape)

    def _create_model(self, input_shape):
        """
        Creates a BiLSTM Autoencoder model.

        :param input_shape: tuple, shape (time_window_size, n_features) for BiLSTM input.
        :return: Compiled BiLSTM Autoencoder model.
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=input_shape))
        model.add(RepeatVector(input_shape[0]))  # The 'time_window_size' dimension is the length of sequences.
        model.add(Bidirectional(LSTM(128, return_sequences=True)))  # Adding a BiLSTM layer for decoding.
        model.add(TimeDistributed(Dense(input_shape[1], activation="linear")))  # Assuming linear activation for reconstruction.

        model.compile(optimizer='adam', loss='mae')  # Assuming Mean Absolute Error (mae) is appropriate.
        return model

    def train(self, X_train, y_train, epochs=60, batch_size=32, validation_split=0.1, shuffle=False):
        """
        Trains the model on the given dataset.

        :param X_train: ndarray, training data.
        :param y_train: ndarray, target values for training data.
        :param epochs: int, number of epochs to train the model.
        :param batch_size: int, number of samples per gradient update.
        :param validation_split: float, fraction of the training data to be used as validation data.
        :param shuffle: boolean, whether to shuffle the training data before each epoch.
        :return: History object, details about the training history at each epoch.
        """
        history = self.model.fit(
            x=X_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle
        )
        return history

    def predict(self, data):
        """
        Generates output predictions for the input samples.

        :param data: ndarray, input data.
        :return: ndarray, predictions.
        """
        predictions = self.model.predict(data)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on the test dataset.

        :param X_test: ndarray, test data.
        :param y_test: ndarray, target values for test data.
        :return: Loss value on the test data.
        """
        loss = self.model.evaluate(X_test, y_test)
        return loss
