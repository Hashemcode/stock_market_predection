import time
import requests

def fetch_and_plot_real_time(ticker):
    while True:
        response = requests.post("http://127.0.0.1:5000/plot", json={"ticker": ticker})
        if response.status_code == 200:
            print(f"{ticker} plot created successfully.")
        else:
            print(f"Failed to create plot for {ticker}: {response.json()}")
        time.sleep(60)  # Fetch and plot every minute

if __name__ == "__main__":
    ticker = "AAPL"  # Replace with the desired ticker symbol
    fetch_and_plot_real_time(ticker)
