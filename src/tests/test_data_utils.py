# src/tests/test_data_utils.py

import pytest
import pandas as pd
from src.data_prep.data_utils import load_data, preprocess_data

def test_load_data():
    """Test loading data from CSV."""
    df = load_data('data/sales_data.csv')
    assert not df.empty
    assert 'Sales' in df.columns

def test_preprocess_data():
    """Test the preprocessing steps."""
    sample_data = pd.DataFrame({
        'Store': [1, 2],
        'Date': ['2023-01-01', '2023-01-02'],
        'Sales': [100, 150]
    })
    processed_data = preprocess_data(sample_data)
    assert pd.api.types.is_datetime64_any_dtype(processed_data['Date'])
    assert processed_data['Store'].dtype.name == 'category'
