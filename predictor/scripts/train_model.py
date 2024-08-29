import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(combined_data_path, model_path, model_type='lstm'):
    if not os.path.exists(combined_data_path):
        print(f"File not found: {combined_data_path}")
        return
    
    data = pd.read_csv(combined_data_path)
    data = data.dropna(subset=['Close', 'support', 'resistance', 'Trendline'])
    
    X = data[['support', 'resistance', 'Trendline']].values
    y = data['Close'].values
    
    X_scaled, scaler_X = preprocess_data(X)
    y_scaled, scaler_y = preprocess_data(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    if model_type == 'lstm':
        model = create_lstm_model((X_train.shape[1], 1))
    elif model_type == 'gru':
        model = create_gru_model((X_train.shape[1], 1))
    elif model_type == 'cnn':
        model = create_cnn_model((X_train.shape[1], 1))
    else:
        print("Invalid model type. Choose 'lstm', 'gru', or 'cnn'.")
        return
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping, model_checkpoint])
    
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_rescaled, y_pred_rescaled)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_test_rescaled - y_pred_rescaled, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.show()

if __name__ == "__main__":
    combined_data_path = "data/combined_data.csv"
    model_path = "models/stock_price_model.keras"
    
    train_model(combined_data_path, model_path, model_type='lstm')  # Change 'lstm' to 'gru' or 'cnn' to train other models