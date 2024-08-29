import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your trained model
model = tf.keras.models.load_model('models/stock_price_model.h5')

# Load your data
data = pd.read_csv("data/combined_data.csv")
data = data.dropna(subset=['Close', 'support', 'resistance', 'Trendline'])

X = data[['support', 'resistance', 'Trendline']].values
y = data['Close'].values

# Preprocess your data
scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
X_scaled = scaler_X.transform(X)
scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1, 1))
y_scaled = scaler_y.transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Calculate metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
