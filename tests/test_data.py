import pytest
import pandas as pd
from src.data.make_dataset import generate_customer_profiles

def test_data_shape():
    """Ensure we generate the correct amount of data."""
    df = generate_customer_profiles(n=100)
    assert len(df) == 100
    assert "customer_id" in df.columns

def test_no_nulls():
    """Ensure our simulation doesn't create broken empty rows."""
    df = generate_customer_profiles(n=50)
    assert df.isnull().sum().sum() == 0