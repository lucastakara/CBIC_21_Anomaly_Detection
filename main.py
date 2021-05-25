from utils.preprocessing import Dataset
from models_architecture.LSTM_Autoencoder import LSTM_Autoencoder
from models_architecture.BILSTM_Autoencoder import BiLSTM_Autoencoder
from models_architecture.CNNLSTM_Autoencoder import CNN_LSTM_Autoencoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def plot_anomaly_points(anomalies, test_score_df, train, test):
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close']])
    train['Close'] = scaler.transform(train[['Close']])
    test['Close'] = scaler.transform(test[['Close']])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=test_score_df['Date'], y=scaler.inverse_transform(test_score_df['Close']), name='Close price'))
    fig.add_trace(
        go.Scatter(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close']), mode='markers', name='Anomaly'))
    #fig.update_layout(showlegend=True, title='LSTM Autoencoder Detected anomalies in IBOVESPA Index')
    fig.show()
    fig.write_image("images/Anomaly_Points.png")
    print(anomalies["Close"].sum())


def plot_train_val_loss(history):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend();
    plt.show()



def determine_anomalies(model, X_train, X_test, test, TIME_STEPS=30):
    X_train_pred = model.predict(X_train)

    # Train MAE Loss
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel('LSTM Autoencoder Train MAE loss')
    plt.ylabel('Number of Samples');
    plt.show()

    threshold = np.max(train_mae_loss)
    print(f'Reconstruction error threshold: {threshold}')

    # Test MAE loss
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    plt.hist(test_mae_loss, bins=50)
    plt.xlabel('LSTM Autoencoder Test MAE loss')
    plt.ylabel('Number of samples');
    plt.show()
    # val_loss 0.0012


    test_score_df = pd.DataFrame(test[TIME_STEPS:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['Close'] = test[TIME_STEPS:]['Close']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['loss'], name='Test loss'))
    fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['threshold'], name='Threshold'))
    fig.update_layout(showlegend=True, title='LSTM Autoencoder Test loss vs. Threshold')
    fig.show()
    fig.write_image("images/fig1.png")

    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

    return test_score_df, anomalies


def main():
    dataset = Dataset()
    df = dataset.get_data
    train, test = dataset.train_test_split(df)
    X_train, y_train, X_test, y_test = dataset.preprocess_df(train, test)
    model = LSTM_Autoencoder(X_train, y_train, X_test, y_test)
    history = model.train(X_train, y_train, epochs=60)
    plot_train_val_loss(history)
    # Evaluate the model on the test data using `evaluate`
    mae = model.evaluate(X_test, y_test)
    print(mae)
    # Model, Train data
    test_score_df, anomalies = determine_anomalies(model, X_train, X_test, test)
    plot_train_val_loss(history)
    plot_anomaly_points(anomalies, test_score_df, train, test)



if __name__ == "__main__":
    main()
