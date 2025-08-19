from pydantic import BaseModel
from typing import List, Optional

class Candle(BaseModel):
    timestamp: str
    price: float

class IndicatorRequest(BaseModel):
    symbol: str
    start: Optional[str] = None  # ISO date/time
    end: Optional[str] = None
    agg: Optional[str] = None    # raw, 1min, 5min, 15min, 1h, 4h, 1d
    # Indicator params (optional)
    sma: Optional[List[int]] = None
    ema: Optional[List[int]] = None
    rsi_period: Optional[int] = 14
    bollinger_period: Optional[int] = 20
    bollinger_std: Optional[float] = 2.0
    macd_fast: Optional[int] = 12
    macd_slow: Optional[int] = 26
    macd_signal: Optional[int] = 9

class SeasonalityResponse(BaseModel):
    symbol: str
    day_of_week: dict
    hour_of_day: dict
