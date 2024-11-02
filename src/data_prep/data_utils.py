# src/data_preprocessing/data_utils.py

import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data."""
    df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
    df['Store'] = df['Store'].astype('category')  # Convert Store to categorical
    return df
