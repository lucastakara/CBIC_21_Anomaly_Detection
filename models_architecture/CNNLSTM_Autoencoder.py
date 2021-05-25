from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv1D, Flatten, MaxPooling1D
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import keras as K

class CNN_LSTM_Autoencoder:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = self.create_model()

    @staticmethod
    def create_model(time_window_size=30, n_features=1):
        model_CNNLSTM = Sequential()
        model_CNNLSTM.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(time_window_size, n_features)))
        model_CNNLSTM.add((MaxPooling1D(pool_size=2)))
        model_CNNLSTM.add(Flatten())
        model_CNNLSTM.add(RepeatVector(time_window_size))
        model_CNNLSTM.add(TimeDistributed(Dense(n_features, activation="linear")))
        model_CNNLSTM.summary()
        plot_model(model_CNNLSTM, to_file="/home/takara/Desktop/Masters/BR_Conference/images/CNN.png", show_shapes=True, show_layer_names=True)
        model_CNNLSTM.compile(optimizer='adam', loss="MAPE")


        return model_CNNLSTM

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
