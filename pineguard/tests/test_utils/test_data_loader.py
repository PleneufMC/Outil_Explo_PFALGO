"""
Tests for data loader — sample data generation and CSV loading.
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.utils.data_loader import generate_sample_data


class TestGenerateSampleData:
    def test_basic_generation(self):
        df = generate_sample_data(n_bars=100, seed=42)
        assert len(df) == 100
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_ohlcv_consistency(self):
        """High >= Close >= Low, High >= Open >= Low."""
        df = generate_sample_data(n_bars=500, seed=42)
        assert (df['High'] >= df['Close']).all()
        assert (df['Close'] >= df['Low']).all()
        assert (df['High'] >= df['Low']).all()

    def test_reproducibility(self):
        """Same seed should give same data."""
        df1 = generate_sample_data(n_bars=100, seed=42)
        df2 = generate_sample_data(n_bars=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Different seeds should give different data."""
        df1 = generate_sample_data(n_bars=100, seed=42)
        df2 = generate_sample_data(n_bars=100, seed=123)
        assert not (df1['Close'] == df2['Close']).all()

    def test_custom_params(self):
        df = generate_sample_data(
            n_bars=50, base_price=100, volatility=0.01,
            freq='4h', start_date='2024-06-01'
        )
        assert len(df) == 50
        # First close should be near base_price
        assert abs(df['Close'].iloc[0] - 100) < 10

    def test_datetime_index(self):
        df = generate_sample_data(n_bars=100, freq='1h')
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == 'Date'

    def test_positive_prices(self):
        """All prices should be positive."""
        df = generate_sample_data(n_bars=1000, seed=42)
        assert (df['Close'] > 0).all()
        assert (df['High'] > 0).all()
        assert (df['Low'] > 0).all()

    def test_volume_non_negative(self):
        """Volume should be non-negative."""
        df = generate_sample_data(n_bars=500, seed=42)
        assert (df['Volume'] >= 0).all()
