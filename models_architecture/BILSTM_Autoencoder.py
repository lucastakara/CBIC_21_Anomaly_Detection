from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional
from tensorflow import keras
from tensorflow.keras.utils import plot_model


class BiLSTM_Autoencoder:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = self.create_model()

    @staticmethod
    def create_model(time_window_size=30, n_features=1):
        model_BiLSTM = Sequential()
        model_BiLSTM.add(Bidirectional(LSTM(units=128, return_sequences=False),
                                       input_shape=(time_window_size, n_features)))
        model_BiLSTM.add(RepeatVector(time_window_size))
        model_BiLSTM.add(TimeDistributed(Dense(n_features, activation="linear")))
        plot_model(model_BiLSTM, to_file="/home/takara/Desktop/Masters/BR_Conference/images/model_BILSTM.png", show_shapes=True, show_layer_names=True)
        model_BiLSTM.compile(optimizer='adam', loss='MAPE')
        model_BiLSTM.summary()

        return model_BiLSTM

    def train(self, X_train, y_train, epochs=60, batch_size=32, validation_split=0.1, shuffle=False):
        history = self.model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
                                  validation_split=validation_split, shuffle=shuffle)
        return history

    def predict(self, X_train):
        X_train_pred = self.model.predict(X_train)
        return X_train_pred

    def evaluate(self, X_test, y_test):
        mae = self.model.evaluate(X_test, y_test)
        return mae
