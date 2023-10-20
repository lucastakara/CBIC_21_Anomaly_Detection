import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from matplotlib.dates import DateFormatter

from utils.preprocessing import Dataset

from models_architectures.LSTM_Autoencoder import LSTMAutoencoder
from models_architectures.BILSTM_Autoencoder import BiLSTMAutoencoder
from models_architectures.CNNLSTM_Autoencoder import CNNLSTMAutoencoder


def plot_anomaly_points(anomalies, test_score_df, train, test, save=False):
    """
    Plots the anomaly points on the graph of stock prices.

    :param anomalies: DataFrame containing information about the anomalies.
    :param test_score_df: DataFrame containing test scores.
    :param train: DataFrame containing the training data.
    :param test: DataFrame containing the test data.
    """
    # Initialize and fit the scaler
    scaler = StandardScaler()
    scaler.fit(train[['Close']])

    # Avoid SettingWithCopyWarning with proper dataframe handling
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled['Close'] = scaler.transform(train[['Close']])
    test_scaled['Close'] = scaler.transform(test[['Close']])

    # Prepare the 'Close' columns for inverse_transform
    test_score_close_reshaped = test_score_df['Close'].values.reshape(-1, 1)
    anomalies_close_reshaped = anomalies['Close'].values.reshape(-1, 1)

    # Create the line plot for close prices
    sns.lineplot(x=test_score_df['Date'], y=scaler.inverse_transform(test_score_close_reshaped).flatten(),
                 color='black', linewidth=1.0)
    sns.scatterplot(x=anomalies['Date'], y=scaler.inverse_transform(anomalies_close_reshaped).flatten(), color='red')

    plt.gcf().autofmt_xdate()  # Rotation for x-axis labels
    plt.gca().set_title('Stock Prices with Anomalies')
    plt.xlabel('Date')  # Setting label for x-axis
    plt.ylabel('Close Price')  # Setting label for y-axis
    plt.tight_layout()

    if save:
        # Check if the directory exists otherwise create it
        save_dir = "images"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)  # Create directory if it does not exist

        # Save the figure
        save_path = os.path.join(save_dir, "Anomaly_Points.png")
        plt.gcf().savefig(save_path)
        print(f"Figure saved to {save_path}")

    # Display the plot
    plt.show()

    # Print the sum of anomaly "Close" values
    print("Sum of anomaly points:", anomalies["Close"].sum())


def plot_train_val_loss(history, save=False):
    """
    Plots the training and validation loss from the model's history.

    :param history: Training history of the model.
    """
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    if save:
        save_dir = "images"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)  # Create directory if it does not exist

        # Save the figure
        save_path = os.path.join(save_dir, "train_val_loss.png")
        plt.gcf().savefig(save_path)
        print(f"Figure saved to {save_path}")

    # Display the plot
    plt.show()


def determine_anomalies(model, X_train, X_test, test, TIME_STEPS=30):
    """
    Determines the anomalies in the reconstructed data based on the model's error.

    :param model: Trained autoencoder model.
    :param X_train: Training data.
    :param X_test: Testing data.
    :param test: Original testing data.
    :param TIME_STEPS: Number of time steps in the sequence.
    """
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel('Train MAE loss')
    plt.ylabel('Number of samples')
    plt.show()

    threshold = np.max(train_mae_loss)
    print(f'Reconstruction error threshold: {threshold}')

    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    sequence_adjustment = TIME_STEPS - 1
    adjusted_test = test.iloc[sequence_adjustment:, :].reset_index(drop=True)

    assert adjusted_test.shape[0] == len(test_mae_loss), "Test set and loss array size mismatch"

    test_score_df = pd.DataFrame(adjusted_test)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > threshold
    test_score_df['Close'] = adjusted_test['Close']

    sns.set(rc={'figure.figsize': (8, 4)})
    sns.set_style("white")
    date_form = DateFormatter("%Y")
    fig = sns.lineplot(x=test_score_df.index, y=test_score_df['threshold'], color='red', linewidth=3)
    fig.xaxis.set_major_formatter(date_form)
    fig = sns.lineplot(x=test_score_df.index, y=test_score_df['loss'], color='black', linewidth=2.5)
    plt.legend(labels=['Threshold line', 'Loss'])
    plt.show()

    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    return test_score_df, anomalies


def main():
    """
    Main function to execute the program.
    """
    dataset = Dataset()
    df = dataset.data
    train, test = dataset.split_data(df)

    X_train, y_train, X_test, y_test = dataset.prepare_data_for_autoencoder(train, test)

    input_shape = (X_train.shape[1], X_train.shape[2])
    for Autoencoder in [LSTMAutoencoder, BiLSTMAutoencoder, CNNLSTMAutoencoder]:
        model = Autoencoder(input_shape)
        history = model.train(X_train, y_train, epochs=2)

        plot_train_val_loss(history, save=True)

        test_score_df, anomalies = determine_anomalies(model, X_train, X_test, test)

        plot_anomaly_points(anomalies, test_score_df, train, test, save=True)


if __name__ == '__main__':
    main()
