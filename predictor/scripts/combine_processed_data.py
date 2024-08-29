import os
import pandas as pd

def combine_processed_data(processed_dir, combined_data_path):
    combined_df = pd.DataFrame()
    
    print("Starting to process files...")
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('_processed.csv'):
            file_path = os.path.join(processed_dir, filename)
            print(f"Processing {filename}...")
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Skipping empty file: {file_path}")
                    continue
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f"Empty data error for file: {file_path}")
            except pd.errors.ParserError:
                print(f"Parsing error for file: {file_path}")
            except Exception as e:
                print(f"Unexpected error processing file {file_path}: {e}")
            print(f"Finished processing {filename}")
    
    if not combined_df.empty:
        combined_df.to_csv(combined_data_path, index=False)
        print(f"Combined data saved to {combined_data_path}")
    else:
        print("No valid data found to combine.")
    
    print("All files processed.")

if __name__ == "__main__":
    processed_dir = "data/processed/"
    combined_data_path = "data/combined_data.csv"
    
    combine_processed_data(processed_dir, combined_data_path)
