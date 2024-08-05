import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the saved model
model = load_model('stock-pred-model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Streamlit app
st.title("StockAlchemy - Predict real time stock prices with ease!")

# User input for stock symbol and date range
stock = st.text_input("Enter Stock Symbol")
start_date = st.date_input("Start Date", pd.to_datetime('2012-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('today'))

# Download stock data
if st.button('Predict'):
    data = yf.download(stock, start_date, end_date)
    data.reset_index(inplace=True)
    
    # Plot moving averages
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(ma_100_days, 'r', label='100 Day MA')
    ax.plot(ma_200_days, 'b', label='200 Day MA')
    ax.plot(data.Close, 'g', label='Close Price')
    ax.legend()
    st.pyplot(fig)

    data.dropna(inplace=True)

    # Split data into training and testing sets
    data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

    scaler.fit(data_train)
    data_train_scale = scaler.transform(data_train)
    
    # Prepare training data for LSTM
    x_train, y_train = [], []
    for i in range(100, data_train_scale.shape[0]):
        x_train.append(data_train_scale[i - 100:i])
        y_train.append(data_train_scale[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Prepare test data
    past_100_days = data_train.tail(100)
    final_df = pd.concat([past_100_days, data_test], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions
    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted)
    
    # Calculate accuracy
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(data_test.values, y_predicted)
    mse = mean_squared_error(data_test.values, y_predicted)
    rmse = np.sqrt(mse)

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot original vs predicted for test data
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data_test.values, 'b', label='Original Prices')
    ax.plot(y_predicted, 'r', label='Predicted Prices')
    ax.set_title('Original vs Predicted Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Predict next 30 days
    last_100_days = input_data[-100:]
    next_30_days_predictions = []
    for _ in range(30):
        next_pred = model.predict(last_100_days.reshape(1, 100, 1))
        next_30_days_predictions.append(next_pred[0, 0])
        last_100_days = np.append(last_100_days[1:], next_pred[0, 0]).reshape(100, 1)

    next_30_days_predictions = scaler.inverse_transform(np.array(next_30_days_predictions).reshape(-1, 1))
    next_30_days_predictions = next_30_days_predictions.flatten()

    # Print the predicted stock prices for the next 30 days
    #st.write("Predicted stock prices for the next 30 days:")
    #st.write(next_30_days_predictions)

    # Plot the next 30 days predictions
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(np.arange(len(data_test), len(data_test) + 30), next_30_days_predictions, 'r', label='Predicted Prices for Next 30 Days')
    ax.set_title('Predicted Stock Prices for the Next 30 Days')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Merge and plot the original data with the next 30 days predictions
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.Close, 'b', label='Original Prices')
    ax.plot(np.arange(len(data), len(data) + 30), next_30_days_predictions, 'r', label='Predicted Prices for Next 30 Days')
    ax.set_title('Original Prices with Predicted Prices for the Next 30 Days')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
