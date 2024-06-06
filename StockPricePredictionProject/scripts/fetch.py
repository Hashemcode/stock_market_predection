import requests
import pandas as pd
import tensorflow as tf

def fetch_predictions(support, resistance, trendline):
    data = {
        "support": [support],
        "resistance": [resistance],
        "Trendline": [trendline]
    }

    response = requests.post("http://127.0.0.1:5000/predict", json=data)

    if response.status_code == 200:
        predictions = response.json().get("predictions")
        return predictions[0]
    else:
        print("Error fetching predictions")
        return None

if __name__ == "__main__":
    support = 1500  # Replace with actual support value
    resistance = 2000  # Replace with actual resistance value
    trendline = 1750  # Replace with actual trendline value
    
    predicted_price = fetch_predictions(support, resistance, trendline)
    
    if predicted_price is not None:
        print(f"Predicted Price: {predicted_price}")
