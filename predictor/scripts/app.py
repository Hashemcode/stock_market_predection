import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__)

# WebSocket and CORS configuration
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8050"}})

# Load the pre-trained model
model = tf.keras.models.load_model('models/stock_price_model.h5')

def calculate_trendline(stock_data):
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Close'].values
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)
    trendline = linear_regressor.predict(X)
    stock_data['Trendline'] = trendline
    return stock_data

@app.route('/price/<ticker>', methods=['GET'])
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period='1d')['Close'].iloc[-1]
        return jsonify({'price': price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if 'ticker' not in data or 'asset_type' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    ticker = data['ticker']
    asset_type = data['asset_type']
    days = int(data.get('days', 30))  # Default to 30 days if period is not provided
    
    # Map days to valid Yahoo Finance periods
    if days <= 5:
        period = '5d'
    elif days <= 30:
        period = '1mo'
    elif days <= 90:
        period = '3mo'
    elif days <= 180:
        period = '6mo'
    elif days <= 365:
        period = '1y'
    else:
        period = 'max'

    try:
        market_data = yf.download(ticker, period=period, interval='1d')
        
        dates = market_data.index.strftime('%Y-%m-%d').tolist()
        close_prices = market_data['Close'].tolist()
        open_prices = market_data['Open'].tolist()
        high_prices = market_data['High'].tolist()
        low_prices = market_data['Low'].tolist()

        support = pd.Series(close_prices).rolling(window=20).min().tolist()
        resistance = pd.Series(close_prices).rolling(window=20).max().tolist()
        market_data = calculate_trendline(market_data)
        trendline = market_data['Trendline'].tolist()

        X = np.array([support, resistance, trendline]).T
        X = X[~np.isnan(X).any(axis=1)]  # Remove rows with NaN values
        scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        predictions = model.predict(X_scaled)
        predictions_rescaled = scaler_X.inverse_transform(np.concatenate([predictions] * 3, axis=1))[:, 0]

        response = {
            'dates': dates[-len(predictions_rescaled):],
            'open': open_prices[-len(predictions_rescaled):],
            'high': high_prices[-len(predictions_rescaled):],
            'low': low_prices[-len(predictions_rescaled):],
            'close': close_prices[-len(predictions_rescaled):],
            'support': support[-len(predictions_rescaled):],
            'resistance': resistance[-len(predictions_rescaled):],
            'trendline': trendline[-len(predictions_rescaled):],
            'predictions': predictions_rescaled.tolist()
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe')
def handle_subscribe(data):
    ticker = data.get('ticker')
    if ticker:
        send_live_data(ticker)

def send_live_data(ticker):
    period = '6mo'
    stock_data = yf.download(ticker, period=period, interval='1d')
    
    if stock_data.empty:
        emit('error', {'error': 'Failed to fetch data'})
        return
    
    stock_data['support'] = stock_data['Close'].rolling(window=20).min()
    stock_data['resistance'] = stock_data['Close'].rolling(window=20).max()
    stock_data = calculate_trendline(stock_data)
    
    X = stock_data[['support', 'resistance', 'Trendline']].dropna().values
    scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_scaled = scaler_X.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    predictions = model.predict(X_scaled)
    
    predictions_rescaled = scaler_X.inverse_transform(np.concatenate([predictions] * 3, axis=1))[:, 0]
    
    support_levels = stock_data['support'].dropna().tolist()
    resistance_levels = stock_data['resistance'].dropna().tolist()
    trendline_levels = stock_data['Trendline'].dropna().tolist()
    prediction_levels = predictions_rescaled.flatten().tolist()
    
    response = {
        'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'close': stock_data['Close'].dropna().tolist(),
        'support': support_levels,
        'resistance': resistance_levels,
        'trendline': trendline_levels,
        'predictions': prediction_levels
    }
    
    emit('live_data', response)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5002, use_reloader=True)
