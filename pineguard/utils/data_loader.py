"""
Data Loading utilities — CSV import, TV trade import, sample data generation.

Supports:
  - MT5 CSV exports (bid prices)
  - TradingView CSV exports (mid prices — preferred)
  - TV Strategy Tester trade exports (Excel/CSV)
  - Sample data generation for testing
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path

DataFrame = pd.DataFrame


def load_csv(
    filepath: str,
    datetime_col: str = 'Date',
    datetime_format: Optional[str] = None,
    timezone: str = 'UTC',
    source: str = 'auto',
) -> DataFrame:
    """
    Load OHLCV data from CSV file.

    Supports MT5 and TradingView CSV formats.

    Args:
        filepath: Path to CSV file
        datetime_col: Column name for datetime (default 'Date')
        datetime_format: Datetime format string (auto-detected if None)
        timezone: Timezone for the data (default 'UTC')
        source: Data source hint: 'mt5', 'tv', or 'auto'

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    """
    df = pd.read_csv(filepath)

    # Auto-detect datetime column
    if datetime_col not in df.columns:
        for candidate in ['Date', 'Time', 'Datetime', 'datetime', 'date', 'time',
                          'Date Time', 'Timestamp', 'timestamp']:
            if candidate in df.columns:
                datetime_col = candidate
                break

    # Parse datetime
    if datetime_format:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    df = df.set_index(datetime_col)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('open', 'o'):
            col_map[col] = 'Open'
        elif cl in ('high', 'h'):
            col_map[col] = 'High'
        elif cl in ('low', 'l'):
            col_map[col] = 'Low'
        elif cl in ('close', 'c'):
            col_map[col] = 'Close'
        elif cl in ('volume', 'vol', 'v'):
            col_map[col] = 'Volume'

    df = df.rename(columns=col_map)

    # Ensure required columns
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # Sort by datetime
    df = df.sort_index()

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def load_tv_trades(
    filepath: str,
    sheet_name: Optional[str] = None,
) -> DataFrame:
    """
    Load TradingView Strategy Tester trade export.

    Pine: load_tv_trades_smart() equivalent.
    TV exports trades as Excel with columns:
      Trade #, Type, Signal, Date/Time, Price, Contracts, Profit, Cum. Profit, etc.

    Args:
        filepath: Path to Excel or CSV file
        sheet_name: Sheet name for Excel files

    Returns:
        DataFrame with standardized trade columns
    """
    ext = Path(filepath).suffix.lower()

    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        df = pd.read_csv(filepath)

    # Standardize columns
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'date' in cl or 'time' in cl:
            col_map[col] = 'datetime'
        elif cl in ('type', 'direction'):
            col_map[col] = 'direction'
        elif cl in ('price', 'entry price'):
            col_map[col] = 'price'
        elif 'profit' in cl and 'cum' not in cl:
            col_map[col] = 'pnl'
        elif 'signal' in cl:
            col_map[col] = 'signal'

    df = df.rename(columns=col_map)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def generate_sample_data(
    n_bars: int = 1000,
    start_date: str = '2024-01-01',
    freq: str = '1h',
    base_price: float = 5000.0,
    volatility: float = 0.001,
    trend: float = 0.00001,
    seed: int = 42,
) -> DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Creates realistic-looking price data with configurable
    volatility and trend. Useful for unit tests and demos.

    Args:
        n_bars: Number of bars to generate
        start_date: Start date string
        freq: Bar frequency ('1h', '4h', '1D', etc.)
        base_price: Starting price level
        volatility: Per-bar volatility (std of returns)
        trend: Per-bar drift (positive = uptrend)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    np.random.seed(seed)

    # Generate returns
    returns = np.random.normal(trend, volatility, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Create OHLC from close prices
    idx = pd.date_range(start=start_date, periods=n_bars, freq=freq)

    close = pd.Series(prices, index=idx)
    noise = np.random.uniform(0.0005, 0.002, n_bars)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close.shift(1).fillna(close.iloc[0])

    # Volume: random with some autocorrelation
    volume = np.abs(np.random.normal(1000, 300, n_bars)).astype(int)

    df = DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=idx)

    df.index.name = 'Date'
    return df
