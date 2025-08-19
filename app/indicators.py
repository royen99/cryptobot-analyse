import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def df_from_series(rows) -> pd.DataFrame:
    # rows: List[(timestamp_iso, price_float)]
    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def add_sma(df: pd.DataFrame, windows: List[int]) -> None:
    for w in windows:
        df[f"SMA_{w}"] = df["price"].rolling(w, min_periods=max(1, w//2)).mean()

def add_ema(df: pd.DataFrame, windows: List[int]) -> None:
    for w in windows:
        df[f"EMA_{w}"] = df["price"].ewm(span=w, adjust=False).mean()

def add_rsi(df: pd.DataFrame, period: int = 14) -> None:
    delta = df["price"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))

def add_bollinger(df: pd.DataFrame, period: int = 20, stds: float = 2.0) -> None:
    ma = df["price"].rolling(period, min_periods=period//2).mean()
    sd = df["price"].rolling(period, min_periods=period//2).std()
    df["BB_MID"] = ma
    df["BB_UPPER"] = ma + stds * sd
    df["BB_LOWER"] = ma - stds * sd

def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> None:
    ema_fast = df["price"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["price"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd.ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

def add_volatility_drawdown(df: pd.DataFrame, window: int = 30) -> None:
    df["RET"] = df["price"].pct_change()
    df["VOL_30"] = df["RET"].rolling(window).std() * np.sqrt(window)
    cummax = df["price"].cummax()
    df["DRAWDOWN"] = df["price"] / cummax - 1.0

def support_resistance(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    # Simple pivot high/low detector on closes
    p = df["price"].values
    piv_high = []
    piv_low = []
    for i in range(len(p)):
        lo = max(0, i - lookback)
        hi = min(len(p), i + lookback + 1)
        seg = p[lo:hi]
        piv_high.append(1 if p[i] == np.max(seg) and len(seg) >= 2 else 0)
        piv_low.append(1 if p[i] == np.min(seg) and len(seg) >= 2 else 0)
    df["PIVOT_HIGH"] = piv_high
    df["PIVOT_LOW"] = piv_low
    return df

def seasonality(df: pd.DataFrame) -> Dict:
    tmp = df.copy()
    tmp["dow"] = tmp["timestamp"].dt.dayofweek
    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["ret"] = tmp["price"].pct_change()
    day_of_week = tmp.groupby("dow")["ret"].mean().dropna().to_dict()
    hour_of_day = tmp.groupby("hour")["ret"].mean().dropna().to_dict()
    # cast keys to str for JSON-friendliness
    return {
        "day_of_week": {str(k): float(v) for k, v in day_of_week.items()},
        "hour_of_day": {str(k): float(v) for k, v in hour_of_day.items()},
    }
