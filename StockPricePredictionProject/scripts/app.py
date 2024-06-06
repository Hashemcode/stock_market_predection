from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Use Agg backend for Matplotlib to avoid GUI issues
matplotlib.use('Agg')

app = Flask(__name__)

model = tf.keras.models.load_model('models/stock_price_model.h5')

def calculate_trendline(stock_data):
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Close'].values
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)
    trendline = linear_regressor.predict(X)
    stock_data['Trendline'] = trendline
    return stock_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if 'ticker' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    ticker = data['ticker']
    period = data.get('period', '6mo')  # Default to 6 months if period is not provided
    
    stock_data = yf.download(ticker, period=period, interval='1d')
    
    if stock_data.empty:
        return jsonify({'error': 'Failed to fetch data'}), 400
    
    stock_data['support'] = stock_data['Close'].rolling(window=20).min()
    stock_data['resistance'] = stock_data['Close'].rolling(window=20).max()
    stock_data = calculate_trendline(stock_data)
    
    X = stock_data[['support', 'resistance', 'Trendline']].dropna().values
    scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_scaled = scaler_X.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    predictions = model.predict(X_scaled)
    
    support_levels = stock_data['support'].dropna().tolist()
    resistance_levels = stock_data['resistance'].dropna().tolist()
    trendline_levels = stock_data['Trendline'].dropna().tolist()
    
    response = {
        'support': support_levels,
        'resistance': resistance_levels,
        'trendline': trendline_levels
    }
    
    return jsonify(response)

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json(force=True)
    
    if 'ticker' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    ticker = data['ticker']
    period = data.get('period', '6mo')  # Default to 6 months if period is not provided
    
    stock_data = yf.download(ticker, period=period, interval='1d')
    
    if stock_data.empty:
        return jsonify({'error': 'Failed to fetch data'}), 400
    
    stock_data['support'] = stock_data['Close'].rolling(window=20).min()
    stock_data['resistance'] = stock_data['Close'].rolling(window=20).max()
    stock_data = calculate_trendline(stock_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['support'], label='Support', color='orange')
    plt.plot(stock_data['resistance'], label='Resistance', color='green')
    plt.plot(stock_data['Trendline'], label='Trendline', color='red')
    
    plt.title(f'{ticker} Stock Price with Support and Resistance ({period})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_stock_price.png')
    plt.close()
    
    return jsonify({'message': f'{ticker} plot created'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
