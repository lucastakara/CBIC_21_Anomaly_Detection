from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed


class CNNLSTMAutoencoder:
    def __init__(self, input_shape):
        """
        Initializes the CNN-LSTM Autoencoder model.

        :param input_shape: tuple, shape (time_window_size, n_features) for CNN input.
        """
        self.model = self._create_model(input_shape)

    @staticmethod
    def _create_model(input_shape):
        """
        Creates a CNN-LSTM Autoencoder model.

        :param input_shape: tuple, shape (time_window_size, n_features) for CNN input.
        :return: Compiled CNN-LSTM Autoencoder model.
        """
        model_CNNLSTM = Sequential()
        model_CNNLSTM.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
        model_CNNLSTM.add(MaxPooling1D(pool_size=2))
        model_CNNLSTM.add(Flatten())
        model_CNNLSTM.add(RepeatVector(input_shape[0]))
        model_CNNLSTM.add(TimeDistributed(Dense(input_shape[1], activation="linear")))

        model_CNNLSTM.compile(optimizer='adam', loss='mae')
        return model_CNNLSTM

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False):
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
