# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
import mplfinance as mpf

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



COMPANY = "TSLA"
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'
PREDICTION_DAYS = 10
PRICE_VALUE = ["Open", "High", "Low", "Close", "Adj Close","Volume"]

# Change features here
features = 'Close'

# Load and process data
original_data, original_train_data, original_test_data, train_data, test_data, scaler = load_and_process_data(
    start_date=TRAIN_START, end_date=TEST_END, company=COMPANY,
    feature=features, split_method="date", test_size=0.2, 
    train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
    save_data=True, load_data=False, na_method='ffill'
)


close_prices_train = train_data[PRICE_VALUE]



# To store the training data
x_train = []
y_train = []

# Prepare the data
for x in range(PREDICTION_DAYS, len(close_prices_train)):
    x_train.append(close_prices_train[x-PREDICTION_DAYS:x])
    y_train.append(close_prices_train.iloc[x, close_prices_train.columns.get_loc('Close')])  # Adjust for target column if needed

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(PRICE_VALUE)))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], len(PRICE_VALUE))))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------


# The above bug is the reason for the following line of code
close_prices_train = train_data[PRICE_VALUE]
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((pd.DataFrame(close_prices_train, columns=PRICE_VALUE), test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(model_inputs)
model_inputs = scaler.transform(model_inputs)
# We need to do the above because to predict the closing price of the first
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

# Note: The following line may not be necessary as model_inputs is already reshaped above.
# model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

# model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(PRICE_VALUE)))
# TO DO: Explain the above 5 lines
# Explanation: The above lines create the test data sequences (x_test) 
# for prediction by taking PREDICTION_DAYS worth of data points for each 
# sequence. x_test is reshaped to fit the input shape of the LSTM model, 
# which is (number of samples, PREDICTION_DAYS, number of features).

predicted_prices = model.predict(x_test)

# Rescale predicted_prices using the close prices scaler
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(total_dataset[['Close']])
predicted_prices = scaler_close.inverse_transform(predicted_prices)

plt.plot(actual_prices[:, PRICE_VALUE.index('Close')], color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

real_data = model_inputs[len(model_inputs) - PREDICTION_DAYS:]
real_data = np.array([real_data])
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], len(PRICE_VALUE)))

prediction = model.predict(real_data)

# Rescale the prediction using the close prices scaler
prediction = scaler_close.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Marketx
# Can you combine these different techniques for a better prediction??
