from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model
import keras

class LSTM_Autoencoder:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = self.create_model()

    @staticmethod
    def create_model(time_window_size=30, n_features=1):
        model = Sequential()
        model.add(LSTM(128, input_shape=(time_window_size, n_features), return_sequences=False))
        model.add(RepeatVector(time_window_size))
        model.add(TimeDistributed(Dense(n_features, activation="linear")))
        plot_model(model, to_file="/home/takara/Desktop/Masters/BR_Conference/images/model_LSTM.png", show_shapes=True, show_layer_names=True)
        model.compile(optimizer='adam', loss='mae')
        model.summary()

        return model

    def train(self, X_train, y_train, epochs=60, batch_size=32, validation_split=0.1, shuffle=False):
        history = self.model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split = validation_split,
                                 shuffle=shuffle)
        return history

    def predict(self, X_train):
        X_train_pred = self.model.predict(X_train)
        return X_train_pred

    def evaluate(self, X_test, y_test):
        mae = self.model.evaluate(X_test, y_test)
        return mae
