"""
PF AI Lab 5.0 — Multi-Format Data Loader
Supports: TradingView CSV, MT5 CSV (tab/semicolon), Excel XLSX
Auto-detects format based on columns/separators.
"""

import pandas as pd
import numpy as np
import os
import warnings


def detect_format(filepath: str) -> str:
    """Auto-detect the data format from the file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return 'excel'

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first_lines = [f.readline() for _ in range(5)]

    header = first_lines[0].lower().strip()

    # TradingView CSV format
    if 'time' in header and 'open' in header and ',' in header:
        return 'tv_csv'

    # MT5 tab-separated
    if '\t' in header and ('<date>' in header or 'date' in header):
        return 'mt5_tab'

    # MT5 semicolon-separated
    if ';' in header and ('date' in header or '<date>' in header):
        return 'mt5_semicolon'

    # Fallback: try comma-separated with OHLC
    if ',' in header:
        return 'generic_csv'

    raise ValueError(
        f"Cannot detect format for {filepath}. "
        "Expected TradingView CSV, MT5 CSV (tab/semicolon), or Excel."
    )


def load_tv_csv(filepath: str) -> pd.DataFrame:
    """Load TradingView exported CSV (mid-price, preferred source)."""
    df = pd.read_csv(filepath)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # TradingView uses 'time' for datetime
    time_col = None
    for candidate in ['time', 'datetime', 'date']:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError("TradingView CSV must have a 'time' column.")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == 'open':
            rename_map[col] = 'Open'
        elif cl == 'high':
            rename_map[col] = 'High'
        elif cl == 'low':
            rename_map[col] = 'Low'
        elif cl == 'close':
            rename_map[col] = 'Close'
        elif cl in ('volume', 'vol'):
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)

    required = ['Open', 'High', 'Low', 'Close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in TradingView CSV.")

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    df.index.name = 'DateTime'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()


def load_mt5_csv(filepath: str, sep: str = '\t') -> pd.DataFrame:
    """Load MT5 exported CSV (bid-price — warn user of divergence)."""
    df = pd.read_csv(filepath, sep=sep)
    df.columns = [c.strip().lower().replace('<', '').replace('>', '') for c in df.columns]

    # Find date and time columns
    date_col = None
    time_col_mt5 = None
    for c in df.columns:
        if 'date' in c:
            date_col = c
        if 'time' in c and 'date' not in c:
            time_col_mt5 = c

    if date_col is None:
        raise ValueError("MT5 CSV must have a 'date' column.")

    if time_col_mt5 and time_col_mt5 in df.columns:
        df['datetime_str'] = df[date_col].astype(str) + ' ' + df[time_col_mt5].astype(str)
    else:
        df['datetime_str'] = df[date_col].astype(str)

    df['DateTime'] = pd.to_datetime(df['datetime_str'])
    df = df.set_index('DateTime')

    rename_map = {}
    volume_mapped = False
    for col in df.columns:
        cl = col.lower()
        if cl == 'open':
            rename_map[col] = 'Open'
        elif cl == 'high':
            rename_map[col] = 'High'
        elif cl == 'low':
            rename_map[col] = 'Low'
        elif cl == 'close':
            rename_map[col] = 'Close'
        elif cl in ('tickvol', 'tick_volume', 'volume', 'vol') and not volume_mapped:
            rename_map[col] = 'Volume'
            volume_mapped = True
    df = df.rename(columns=rename_map)

    # Drop duplicate columns (e.g., MT5 has both tickvol and vol)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    required = ['Open', 'High', 'Low', 'Close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in MT5 CSV.")

    return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()


def load_excel(filepath: str) -> pd.DataFrame:
    """Load Excel file (pandas read_excel)."""
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Try to find datetime column
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ('time', 'datetime', 'date', 'date/time'):
            time_col = c
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == 'open':
            rename_map[col] = 'Open'
        elif cl == 'high':
            rename_map[col] = 'High'
        elif cl == 'low':
            rename_map[col] = 'Low'
        elif cl == 'close':
            rename_map[col] = 'Close'
        elif cl in ('volume', 'vol'):
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    required = ['Open', 'High', 'Low', 'Close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column '{r}' in Excel file.")

    df.index.name = 'DateTime'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()


def load_generic_csv(filepath: str) -> pd.DataFrame:
    """Fallback loader for comma-separated CSV files."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    time_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ('time', 'datetime', 'date', 'date/time', 'timestamp'):
            time_col = c
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    rename_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == 'open':
            rename_map[col] = 'Open'
        elif cl == 'high':
            rename_map[col] = 'High'
        elif cl == 'low':
            rename_map[col] = 'Low'
        elif cl == 'close':
            rename_map[col] = 'Close'
        elif cl in ('volume', 'vol'):
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    required = ['Open', 'High', 'Low', 'Close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column '{r}' in CSV file.")

    df.index.name = 'DateTime'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()


def load_data(filepath: str, source_hint: str = None) -> tuple:
    """
    Main entry point: auto-detect format and load OHLCV data.

    Returns:
        (DataFrame, source_type: str)
        source_type in {'tv_csv', 'mt5_tab', 'mt5_semicolon', 'excel', 'generic_csv'}
    """
    if source_hint:
        fmt = source_hint
    else:
        fmt = detect_format(filepath)

    loaders = {
        'tv_csv': load_tv_csv,
        'mt5_tab': lambda fp: load_mt5_csv(fp, sep='\t'),
        'mt5_semicolon': lambda fp: load_mt5_csv(fp, sep=';'),
        'excel': load_excel,
        'generic_csv': load_generic_csv,
    }

    if fmt not in loaders:
        raise ValueError(f"Unknown format: {fmt}")

    df = loaders[fmt](filepath)

    # Drop any remaining duplicate columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Convert to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            # Ensure it's a Series not a DataFrame (safety check)
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Add hl2 source column (used by PMax)
    df['hl2'] = (df['High'] + df['Low']) / 2.0

    # Warn if MT5 data
    if fmt in ('mt5_tab', 'mt5_semicolon'):
        warnings.warn(
            "\n⚠️  MT5 data detected (bid-price). "
            "Expect 0.5-2.0 points divergence vs TradingView (mid-price). "
            "Results are EXPLORATORY — validate in TradingView.\n",
            UserWarning
        )

    return df, fmt


def load_tv_trades(filepath: str) -> tuple:
    """
    Load TradingView Strategy Tester export (Excel/CSV).
    Expected columns: Trade #, Type, Signal, Date/Time, Price, Profit, % du PnL
    Flexible column matching.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = [c.strip() for c in df.columns]

    # Try to find key columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'trade' in cl and '#' in cl:
            col_map['trade_num'] = c
        elif cl in ('type',):
            col_map['type'] = c
        elif 'signal' in cl:
            col_map['signal'] = c
        elif 'date' in cl or 'time' in cl:
            if 'date' not in col_map:
                col_map['date'] = c
        elif cl == 'price':
            col_map['price'] = c
        elif 'profit' in cl and '%' not in cl and 'pnl' not in cl:
            col_map['profit'] = c
        elif 'profit' in cl and ('%' in cl or 'pnl' in cl):
            col_map['profit_pct'] = c
        elif cl in ('contracts', 'qty', 'quantity'):
            col_map['qty'] = c

    return df, col_map


def load_tv_trades_smart(filepath: str) -> tuple:
    """
    Smart loader for TradingView Strategy Tester exports.

    Handles the multi-sheet Excel format exported by TradingView:
      - Sheet 'List of trades' contains Entry+Exit rows (2 rows per trade)
      - Only Exit rows carry the actual P&L for each trade
      - Falls back to first sheet for simple CSV / single-sheet Excel

    Returns:
        (pnls_pct: np.ndarray, pnls_usd: np.ndarray, n_total_trades: int, info: dict)
        where pnls_pct is the preferred P&L array (percentages),
        pnls_usd is the absolute P&L array,
        n_total_trades is the count of unique trades,
        info contains metadata about parsing.
    """
    ext = os.path.splitext(filepath)[1].lower()
    df = None
    sheet_used = 'default'

    # --- Step 1: Load the correct sheet ---
    if ext in ('.xlsx', '.xls'):
        try:
            xls = pd.ExcelFile(filepath)
            sheet_names = xls.sheet_names

            # TradingView multi-sheet export: look for 'List of trades'
            trades_sheet = None
            for s in sheet_names:
                if 'list of trades' in s.lower():
                    trades_sheet = s
                    break
            # Fallback: look for sheet with 'trades' in name
            if trades_sheet is None:
                for s in sheet_names:
                    if 'trade' in s.lower():
                        trades_sheet = s
                        break

            if trades_sheet:
                df = pd.read_excel(xls, sheet_name=trades_sheet)
                sheet_used = trades_sheet
            else:
                df = pd.read_excel(xls, sheet_name=0)
                sheet_used = sheet_names[0] if sheet_names else 'default'
        except Exception:
            df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        sheet_used = 'csv'

    df.columns = [c.strip() for c in df.columns]

    # --- Step 2: Detect column mapping ---
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'trade' in cl and '#' in cl:
            col_map['trade_num'] = c
        elif cl == 'type':
            col_map['type'] = c
        # P&L percentage columns (prioritize these)
        elif ('p&l' in cl or 'pnl' in cl or 'profit' in cl) and '%' in cl:
            if 'net' in cl or 'pnl_pct' not in col_map:
                col_map['pnl_pct'] = c
        # P&L absolute columns
        elif ('p&l' in cl or 'pnl' in cl or 'profit' in cl) and '%' not in cl:
            if 'cumulative' not in cl and 'favorable' not in cl and 'adverse' not in cl:
                if 'net' in cl or 'pnl_usd' not in col_map:
                    col_map['pnl_usd'] = c

    # --- Step 3: Filter to Exit rows only (avoid double-counting) ---
    if 'type' in col_map:
        type_col = col_map['type']
        # TradingView uses 'Exit long', 'Exit short', 'Entry long', 'Entry short'
        exit_mask = df[type_col].astype(str).str.lower().str.contains('exit', na=False)
        entry_mask = df[type_col].astype(str).str.lower().str.contains('entry', na=False)

        if exit_mask.any() and entry_mask.any():
            # Multi-row format: only keep Exit rows
            df_trades = df[exit_mask].copy()
        else:
            # Single-row format or no Entry/Exit distinction
            df_trades = df.copy()
    else:
        df_trades = df.copy()

    # --- Step 4: Extract P&L arrays ---
    pnls_pct = np.array([])
    pnls_usd = np.array([])

    if 'pnl_pct' in col_map:
        pnls_pct = pd.to_numeric(df_trades[col_map['pnl_pct']], errors='coerce').dropna().values
    if 'pnl_usd' in col_map:
        pnls_usd = pd.to_numeric(df_trades[col_map['pnl_usd']], errors='coerce').dropna().values

    # Fallback: if no P&L columns found, try numeric detection
    if len(pnls_pct) == 0 and len(pnls_usd) == 0:
        for c in df_trades.columns:
            cl = c.lower()
            if 'trade' in cl or 'type' in cl or 'signal' in cl or 'date' in cl or 'time' in cl:
                continue
            vals = pd.to_numeric(df_trades[c], errors='coerce').dropna()
            if len(vals) >= 5 and vals.std() > 0:
                pnls_usd = vals.values
                col_map['pnl_usd_fallback'] = c
                break

    # Count unique trades
    if 'trade_num' in col_map:
        n_total = df_trades[col_map['trade_num']].nunique()
    else:
        n_total = max(len(pnls_pct), len(pnls_usd))

    info = {
        'sheet_used': sheet_used,
        'total_rows': len(df),
        'exit_rows': len(df_trades),
        'col_map': col_map,
        'columns': list(df.columns),
        'has_pct': len(pnls_pct) > 0,
        'has_usd': len(pnls_usd) > 0,
    }

    return pnls_pct, pnls_usd, n_total, info
