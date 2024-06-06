import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to preprocess data
def preprocess_data(raw_data_path, processed_data_path):
    try:
        data = pd.read_csv(raw_data_path)
        data = data.ffill()  # Use .ffill() to forward-fill NaN values
        data.to_csv(processed_data_path, index=False)
    except pd.errors.EmptyDataError:
        print(f"Empty data error for file: {raw_data_path}")
    except pd.errors.ParserError:
        print(f"Parsing error for file: {raw_data_path}")
    except Exception as e:
        print(f"Unexpected error processing file {raw_data_path}: {e}")

# Function to detect support and resistance
def detect_support_resistance(df, window=5):
    try:
        df['support'] = df['Low'].rolling(window, center=True).min()
        df['resistance'] = df['High'].rolling(window, center=True).max()
    except KeyError:
        print("Required columns for support and resistance not found in dataframe")
    except Exception as e:
        print(f"Unexpected error in detect_support_resistance: {e}")
    return df

# Function to detect trendlines
def detect_trendlines(df):
    try:
        x = np.arange(len(df))
        y = df['Close'].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        
        trendline = model.predict(np.arange(len(df)).reshape(-1, 1))
        df['Trendline'] = trendline  # Assign the trendline directly
    except KeyError:
        print("Required column 'Close' not found in dataframe")
    except Exception as e:
        print(f"Unexpected error in detect_trendlines: {e}")
    return df

# Function to generate figures
def generate_figures(data_path, save_path):
    try:
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.plot(data['Date'], data['support'], label='Support')
        plt.plot(data['Date'], data['resistance'], label='Resistance')
        plt.plot(data['Date'], data['Trendline'], label='Trendline')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    except KeyError:
        print("Required columns for plotting not found in dataframe")
    except Exception as e:
        print(f"Unexpected error in generate_figures: {e}")

# Main function to process all files
def process_all_files(raw_dir, processed_dir, figure_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    for filename in os.listdir(raw_dir):
        if filename.endswith('.csv'):
            ticker = filename.split('.')[0]
            raw_data_path = os.path.join(raw_dir, filename)
            processed_data_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
            figure_path = os.path.join(figure_dir, f"{ticker}_stock_price_analysis.png")
            
            print(f"Processing {ticker}...")
            
            # Preprocess data
            preprocess_data(raw_data_path, processed_data_path)
            
            # Detect support, resistance, and trendlines
            try:
                data = pd.read_csv(processed_data_path)
                data = detect_support_resistance(data)
                data = detect_trendlines(data)
                data.to_csv(processed_data_path, index=False)
                
                # Generate figures
                generate_figures(processed_data_path, figure_path)
            except Exception as e:
                print(f"Unexpected error reading processed file {processed_data_path}: {e}")

if __name__ == "__main__":
    raw_dir = "data/raw/"
    processed_dir = "data/processed/"
    figure_dir = "docs/figures/"
    
    process_all_files(raw_dir, processed_dir, figure_dir)
