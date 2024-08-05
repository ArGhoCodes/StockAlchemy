# StockAlchemy
StockAlchemy is an advanced stock price prediction model built using Long Short-Term Memory (LSTM) networks. This model leverages historical stock price data to forecast future stock prices, providing valuable insights for investors and analysts.

Features
LSTM-based Forecasting: Utilizes LSTM networks to capture temporal dependencies and trends in stock price data.
Moving Average Analysis: Includes moving average calculations to help identify trends and potential buy/sell signals.
Model Evaluation: Provides metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate model performance.
Future Predictions: Predicts stock prices for the next 30 days based on historical data.
Interactive Visualization: Visualizes stock prices, predictions, and performance metrics.  

Here's a detailed video explanation on how StockAlchemy works:

https://github.com/user-attachments/assets/31c9d664-fc14-4b2c-8641-7af7c5f9a009


Installation
To use StockAlchemy, clone this repository and install the required dependencies.

Clone the Repository

bash

git clone https://github.com/yourusername/StockAlchemy.git

cd StockAlchemy

Install Dependencies

Ensure you have Python 3.6+ installed. Then, install the necessary packages:

bash

pip install -r requirements.txt

Usage
Prepare the Data: Ensure you have historical stock price data available. The script uses Yahoo Finance to download data, but you can modify it to use other sources.

Train the Model: Run the script to train the LSTM model. The model will be saved as stock-pred-model.h5.

Run the Streamlit App: Start the Streamlit app to interact with the model and visualize predictions.

bash

streamlit run app.py

Model Performance

The model’s performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). The lower these values, the better the model’s performance.

Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

Keras for the LSTM implementation.

Yahoo Finance for stock data.

Streamlit for creating interactive web applications

For any queries, drop a mail at argho202@gmail.com
