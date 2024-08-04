#pip install numpy matplotlib pandas yfinance scikit-learn tensorflow mplfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
import mplfinance as mpf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


# Data Config
COMPANY = "TSLA"
TRAIN_START = '2015-01-01'
TRAIN_END = '2016-05-01'
TEST_START = '2016-05-01'
TEST_END = '2018-12-31'
PREDICTION_DAYS = 10
FEATURE = "Close"


# Define layer Config
layer_configs = [
    {'layer_type': 'LSTM', 'num_layers': 2, 'layer_size': 50},
    {'layer_type': 'GRU', 'num_layers': 2, 'layer_size': 50},
    {'layer_type': 'RNN', 'num_layers': 2, 'layer_size': 50},
]

def load_and_process_data(start_date, end_date, company="TSLA",
                          feature="Close", split_method="ratio", test_size=0.2,
                          train_start=None, train_end=None, test_start=None, test_end=None,
                          save_data=False, load_data=False, na_method='ffill'):
    # Check if loading from local data is requested
    if load_data and os.path.exists("stock_data.csv"):
        data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
    else:
        # Load data from source
        data = yf.download(company, start=start_date, end=end_date, progress=False)

        # Save data if requested
        if save_data:
            data.to_csv("stock_data.csv")

    # Print the columns of the loaded data
    print("Available columns in the data:", data.columns)

    # Keep a copy of the original data for visualization
    original_data = data.copy()

    # Handle NaN values
    if na_method == 'ffill':
        data.fillna(method='ffill', inplace=True)
    elif na_method == 'bfill':
        data.fillna(method='bfill', inplace=True)
    elif na_method == 'drop':
        data.dropna(inplace=True)
    elif na_method == 'zero':
        data.fillna(0, inplace=True)

    # Scale data for the selected feature
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    # Print scaled data
    print(f"Scaled data for {feature}:\n", data[feature].head())

    # Split scaled data into train and test sets
    if split_method == "ratio":
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
        original_train_data, original_test_data = original_data[:split_index], original_data[split_index:]
    elif split_method == "date":
        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]
        original_train_data = original_data.loc[train_start:train_end]
        original_test_data = original_data.loc[test_start:test_end]

    # Return original and scaled data, train/test data, and scaler
    return original_data, original_train_data, original_test_data, train_data, test_data, scaler

# Load and process data
original_data, original_train_data, original_test_data, train_data, test_data, scaler = load_and_process_data(
    start_date=TRAIN_START, end_date=TEST_END, company=COMPANY,
    feature=FEATURE, split_method="date", test_size=0.2, 
    train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
    save_data=True, load_data=False, na_method='ffill'
)


def create_candlestick_chart(dataset, company):
    # Define the custom style
    custom_style = mpf.make_mpf_style(
        base_mpf_style='classic',
        marketcolors=mpf.make_marketcolors(
            up='blue', down='red', edge='inherit', wick='inherit', volume='in'
        )
    )

    # Plot the candlestick chart with the custom style
    mpf.plot(
        dataset,
        type='candle',
        style=custom_style,
        title=f"{company} Stock Price",
        ylabel=f"{company} Stock Price",
        ylabel_lower=f"{company} Volume"
    )
create_candlestick_chart(original_test_data,COMPANY)

def plot_boxplots(ticker, start_date, end_date, save_chart=False, chart_path='chart/multiple_boxplots_chart.png'):
    # Fetch the stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    # Create a boxplot for the 'Open', 'High', 'Low', 'Close', and 'Adj Close' prices
    plt.figure(figsize=(12, 8))
    df_to_plot = df[['Open', 'High', 'Low', 'Close', 'Adj Close']]
    boxplot = df_to_plot.boxplot(patch_artist=True)
    
    # Set the color of the boxes
    for box in boxplot.artists:
        box.set_facecolor('blue')
    
    plt.title(f'{ticker} Boxplot Chart ({start_date} to {end_date})')
    plt.ylabel('Price ($)')
    plt.xticks([1, 2, 3, 4, 5], ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Adj Close'])
 
    # Show the plot
    plt.show()

plot_boxplots(ticker="AAPL", start_date="2023-01-01", end_date="2023-12-31")   

def prepare_training_data(data, prediction_days=60, feature_col='Close'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_col].values.reshape(-1, 1))
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

x_train, y_train, scaler = prepare_training_data(train_data, PREDICTION_DAYS)
x_test, y_test, _ = prepare_training_data(test_data, PREDICTION_DAYS)

def create_dl_model(num_layers, layer_size, layer_type):
    model = Sequential()

    for i in range(num_layers):
        if i == 0:
            if layer_type == 'LSTM':
                model.add(LSTM(layer_size, return_sequences=True, input_shape=(None, 1)))
            elif layer_type == 'GRU':
                model.add(GRU(layer_size, return_sequences=True, input_shape=(None, 1)))
            elif layer_type == 'RNN':
                model.add(SimpleRNN(layer_size, return_sequences=True, input_shape=(None, 1)))
        else:
            if layer_type == 'LSTM':
                model.add(LSTM(layer_size, return_sequences=(i < num_layers - 1)))
            elif layer_type == 'GRU':
                model.add(GRU(layer_size, return_sequences=(i < num_layers - 1)))
            elif layer_type == 'RNN':
                model.add(SimpleRNN(layer_size, return_sequences=(i < num_layers - 1)))
        model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model

def train_and_plot_model(model, X_train, y_train, X_test, y_test, scaler, epochs=25, batch_size=32, model_name="model"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_stock_price = scaler.inverse_transform(y_train.reshape(-1, 1))

    plt.figure(figsize=(14, 7))
    plt.plot(range(len(train_stock_price), len(train_stock_price) + len(real_stock_price)), real_stock_price, color='red', label='Real Stock Price')
    plt.plot(range(len(train_stock_price), len(train_stock_price) + len(predicted_stock_price)), predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title(f'{model_name} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Train and plot for each model
for config in layer_configs:
    model = create_dl_model(config['num_layers'], config['layer_size'], config['layer_type'])
    model_name = f"{config['layer_type']}_{config['num_layers']}_layers"
    train_and_plot_model(model, x_train, y_train, x_test, y_test, scaler, epochs=25, batch_size=32, model_name=model_name)