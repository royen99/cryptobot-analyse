# app/config.py
from __future__ import annotations
import json, os, logging
from functools import lru_cache
from typing import Optional, Dict
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class CoinPrecision(BaseModel):
    price: int = 4
    amount: int = 6

class CoinMinOrder(BaseModel):
    buy: float | int | None = None
    sell: float | int | None = None

class CoinSettings(BaseModel):
    enabled: bool = False
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    rebuy_discount: Optional[float] = None
    volatility_window: Optional[int] = None
    trend_window: Optional[int] = None
    macd_short_window: Optional[int] = 12
    macd_long_window: Optional[int] = 26
    macd_signal_window: Optional[int] = 9
    rsi_period: Optional[int] = 14
    trail_percent: Optional[float] = None
    min_order_sizes: CoinMinOrder = CoinMinOrder()
    precision: CoinPrecision = CoinPrecision()

class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[int] = None

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "trading"
    user: str = "postgres"
    password: str = "postgres"

class RootConfig(BaseModel):
    name: Optional[str] = None
    privateKey: Optional[str] = None
    trade_percentage: Optional[float] = None
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    buy_offset_percent: Optional[float] = None
    sell_offset_percent: Optional[float] = None
    stop_loss_percentage: Optional[float] = None
    trail_percent: Optional[float] = None
    telegram: TelegramConfig = TelegramConfig()
    database: DatabaseConfig = DatabaseConfig()
    coins: Dict[str, CoinSettings] = {}

@lru_cache(maxsize=1)
def get_config() -> RootConfig:
    """
    Load configuration from CONFIG_PATH (JSON). If not found, fall back to env vars for DB only.
    NOTE: We never log secrets.
    """
    path = os.getenv("CONFIG_PATH", "/config/config.json")
    log = logging.getLogger("config")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = RootConfig.model_validate(data)
        log.info(
            "Loaded config.json from %s (db=%s@%s:%s/%s)",
            path, cfg.database.user, cfg.database.host, cfg.database.port, cfg.database.name
        )
        return cfg
    except FileNotFoundError:
        log.warning("CONFIG_PATH %s not found; falling back to DB_* environment vars.", path)
        db = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "trading"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
        )
        return RootConfig(database=db)
    except Exception as e:
        log.exception("Failed to load config from %s", path)
        # Bubble up; FastAPI will show 500 until fixed (better than using wrong creds)
        raise

def get_enabled_symbols() -> list[str]:
    cfg = get_config()
    return [sym for sym, c in (cfg.coins or {}).items() if c.enabled]

def get_coin_settings(symbol: str) -> Optional[CoinSettings]:
    return (get_config().coins or {}).get(symbol)

# ---- App-level knobs still via env (unchanged) ----
class AppSettings(BaseSettings):
    DEFAULT_AGG: str = "15min"
    MAX_LOOKBACK_DAYS: int = 1095

    class Config:
        env_file = ".env"  # optional; ignored if not present

settings = AppSettings()
