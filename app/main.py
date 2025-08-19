from dataclasses import asdict
from fastapi import FastAPI, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timedelta
import io, csv, os
from typing import Optional, List
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .config import settings, get_config, get_enabled_symbols, get_coin_settings
from .db import get_session
from .queries import minute_bucket_sql
from .utils import to_float_rows
from .indicators import (
    df_from_series, add_sma, add_ema, add_rsi, add_bollinger, add_macd,
    add_volatility_drawdown, support_resistance, seasonality
)
from .analysis import compute_drawdown_events, dataframe_from_series, compute_optimal_drop_grid, current_recommendation

app = FastAPI(title="Crypto Long-Term Analyzer")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOW_ANY_SYMBOL = os.getenv("ALLOW_ANY_SYMBOL", "0") == "1"

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

def clamp_lookback(start: Optional[str], max_days: int) -> Optional[str]:
    # Very light guard; you can hard-enforce server-side if you want
    return start

def parse_start_end(start: str | None, end: str | None):
    sdt = edt = None
    if start:
        sdt = datetime.strptime(start, "%Y-%m-%d") if len(start) == 10 else datetime.fromisoformat(start)
    if end:
        edt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)) if len(end) == 10 else datetime.fromisoformat(end)
    return sdt, edt

async def ensure_symbol_enabled(symbol: str):
    if ALLOW_ANY_SYMBOL:
        return symbol
    if symbol not in get_enabled_symbols():
        raise HTTPException(status_code=400, detail=f"Symbol '{symbol}' is not enabled in config.json")
    return symbol

@app.get("/api/config/coins")
async def list_enabled_coins():
    cfg = get_config()
    enabled = [s for s, c in (cfg.coins or {}).items() if c.enabled]
    # Return only coin-level settings (no secrets)
    return {
        "enabled": enabled,
        "coins": {sym: c.model_dump() for sym, c in (cfg.coins or {}).items()}
    }

@app.get("/api/lt/candles")
async def candles(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    agg = agg or settings.DEFAULT_AGG
    bucket = minute_bucket_sql(agg)

    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')
        params["end"] = end_dt

    sql = (
        f"""
        SELECT {bucket} AS ts, avg(price) AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        GROUP BY ts
        ORDER BY ts ASC
        """
        if bucket
        else
        f"""
        SELECT "timestamp" AS ts, price
        FROM price_history
        WHERE {' AND '.join(where)}
        ORDER BY "timestamp" ASC
        """
    )

    rows = (await session.execute(text(sql), params)).all()
    data = to_float_rows(rows)
    return JSONResponse({
        "symbol": symbol,
        "agg": agg,
        "points": [{"timestamp": ts, "price": pr} for ts, pr in data]
    })


@app.get("/api/lt/latest")
async def latest_ticks(
    symbol: str = Depends(ensure_symbol_enabled),
    limit: int = 5000,  # tune as you like
    session: AsyncSession = Depends(get_session),
):
    # CTE for fast DESC scan, then re-order ASC for charting
    sql = """
        WITH latest AS (
            SELECT "timestamp" AS ts, price
            FROM price_history
            WHERE symbol = :symbol
            ORDER BY "timestamp" DESC
            LIMIT :limit
        )
        SELECT ts, price FROM latest ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), {"symbol": symbol, "limit": limit})).all()
    data = to_float_rows(rows)
    return JSONResponse({
        "symbol": symbol,
        "points": [{"timestamp": ts, "price": pr} for ts, pr in data]
    })

@app.get("/api/lt/indicators")
async def indicators(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = None,
    sma: Optional[List[int]] = Query(None),
    ema: Optional[List[int]] = Query(None),
    rsi_period: int = 14,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    session: AsyncSession = Depends(get_session),
):
    agg = agg or settings.DEFAULT_AGG
    bucket = minute_bucket_sql(agg)

    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')  # half-open
        params["end"] = end_dt

    sql = f"""
        SELECT {bucket if bucket else '"timestamp"'} AS ts,
               {"avg(price)" if bucket else "price"} AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        {f"GROUP BY ts" if bucket else ""}
        ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), params)).all()
    series = to_float_rows(rows)
    df = df_from_series(series)

    # indicators
    if sma:
        add_sma(df, sma)
    if ema:
        add_ema(df, ema)
    add_rsi(df, rsi_period)
    add_bollinger(df, bollinger_period, bollinger_std)
    add_macd(df, macd_fast, macd_slow, macd_signal)
    add_volatility_drawdown(df, 30)
    support_resistance(df, 5)

    df = df.bfill().ffill()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    records = df.to_dict(orient="records")
    return JSONResponse({"symbol": symbol, "agg": agg, "data": records})

@app.get("/api/lt/seasonality")
async def seasonality_view(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = "1h",
    session: AsyncSession = Depends(get_session),
):
    bucket = minute_bucket_sql(agg)
    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')  # half-open
        params["end"] = end_dt

    sql = f"""
        SELECT {bucket} AS ts, avg(price) AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        GROUP BY ts
        ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), params)).all()
    df = df_from_series(to_float_rows(rows))
    se = seasonality(df)
    return JSONResponse({"symbol": symbol, **se})

@app.get("/api/lt/export.csv")
async def export_csv(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    agg = agg or settings.DEFAULT_AGG
    bucket = minute_bucket_sql(agg)

    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')  # half-open
        params["end"] = end_dt

    sql = f"""
        SELECT {bucket if bucket else '"timestamp"'} AS ts,
               {"avg(price)" if bucket else "price"} AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        {f"GROUP BY ts" if bucket else ""}
        ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), params)).all()
    data = to_float_rows(rows)

    def gen():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "price"])
        for ts, pr in data:
            writer.writerow([ts, pr])
        yield buf.getvalue()

    return StreamingResponse(gen(), media_type="text/csv")

@app.get("/api/lt/drawdowns")
async def drawdowns_view(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = "1h",          # hourly by default for stability
    min_drop: float = 0.10,             # -10% or worse
    recovery_tolerance: float = 0.0,    # accept full recovery to prior peak (0.0) or within X (e.g. 0.005)
    lookahead_days: Optional[int] = 60, # give up after 60 days if no recovery
    overshoot_window_days: int = 14,    # measure 2 weeks after recovery
    session: AsyncSession = Depends(get_session),
):
    """
    Finds historical drop->recovery events and summarizes recovery behavior.
    - Returns both events and summary stats.
    """
    bucket = minute_bucket_sql(agg)
    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')
        params["end"] = end_dt

    # Pull series (aggregate if bucketed)
    sql = f"""
        SELECT {bucket if bucket else '"timestamp"'} AS ts,
               {"avg(price)" if bucket else "price"} AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        {f"GROUP BY ts" if bucket else ""}
        ORDER BY ts ASC
    """

    rows = (await session.execute(text(sql), params)).all()
    series = to_float_rows(rows)  # -> [(iso, float), ...]

    df = dataframe_from_series(series)
    if df.empty:
        return JSONResponse({
            "symbol": symbol,
            "agg": agg,
            "events": [],
            "summary": {
                "avg_recovery_days": None,
                "median_recovery_days": None,
                "success_rate": None,
                "worst_recovery_days": None,
                "events": 0
            }
        })

    events, summary = compute_drawdown_events(
        df,
        min_drop=min_drop,
        recovery_tolerance=recovery_tolerance,
        lookahead_days=lookahead_days,
        overshoot_window_days=overshoot_window_days,
    )

    # JSON-ify dataclasses
    events_json = [asdict(e) for e in events]

    return JSONResponse({
        "symbol": symbol,
        "agg": agg,
        "params": {
            "min_drop": min_drop,
            "recovery_tolerance": recovery_tolerance,
            "lookahead_days": lookahead_days,
            "overshoot_window_days": overshoot_window_days,
        },
        "events": events_json,
        "summary": summary
    })

@app.get("/api/lt/optimal_drop")
async def optimal_drop_view(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = "1h",
    target_pct: float = 0.05,
    drop_min: float = 0.05,
    drop_max: float = 0.40,
    drop_step: float = 0.01,
    lookahead_days: int = 60,
    score_mode: str = "expected_return_per_year",  # success_rate | successes_per_year | expected_return_per_year | sr_over_median_days
    session: AsyncSession = Depends(get_session),
):
    bucket = minute_bucket_sql(agg)
    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')
        params["end"] = end_dt

    sql = f"""
        SELECT {bucket if bucket else '"timestamp"'} AS ts,
               {"avg(price)" if bucket else "price"} AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        {f"GROUP BY ts" if bucket else ""}
        ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), params)).all()
    series = to_float_rows(rows)
    df = dataframe_from_series(series)
    if df.empty:
        return JSONResponse({
            "symbol": symbol,
            "agg": agg,
            "target_pct": target_pct,
            "params": {
                "drop_min": drop_min, "drop_max": drop_max, "drop_step": drop_step,
                "lookahead_days": lookahead_days, "score_mode": score_mode
            },
            "grid": [],
            "best": None
        })

    grid, best = compute_optimal_drop_grid(
        df,
        target_pct=target_pct,
        drop_min=drop_min,
        drop_max=drop_max,
        drop_step=drop_step,
        lookahead_days=lookahead_days,
        score_mode=score_mode,
    )

    return JSONResponse({
        "symbol": symbol,
        "agg": agg,
        "target_pct": target_pct,
        "params": {
            "drop_min": drop_min, "drop_max": drop_max, "drop_step": drop_step,
            "lookahead_days": lookahead_days, "score_mode": score_mode
        },
        "grid": grid,
        "best": best
    })

@app.get("/api/lt/recommendation")
async def recommendation_view(
    symbol: str = Depends(ensure_symbol_enabled),
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = "1h",
    target_pct: float = 0.05,
    drop_min: float = 0.05,
    drop_max: float = 0.40,
    drop_step: float = 0.01,
    lookahead_days: int = 60,
    score_mode: str = "expected_return_per_year",
    peak_lookback_days: int = 180,
    session: AsyncSession = Depends(get_session),
):
    """
    Uses optimal_drop grid to decide: buy now vs wait, given the current price and recent peak.
    """
    # fetch series
    bucket = minute_bucket_sql(agg)
    where = ['symbol = :symbol']
    params = {"symbol": symbol}

    start_dt, end_dt = parse_start_end(start, end)
    if start_dt:
        where.append('"timestamp" >= :start')
        params["start"] = start_dt
    if end_dt:
        where.append('"timestamp" < :end')
        params["end"] = end_dt

    sql = f"""
        SELECT {bucket if bucket else '"timestamp"'} AS ts,
               {"avg(price)" if bucket else "price"} AS price
        FROM price_history
        WHERE {' AND '.join(where)}
        {f"GROUP BY ts" if bucket else ""}
        ORDER BY ts ASC
    """
    rows = (await session.execute(text(sql), params)).all()
    series = to_float_rows(rows)
    df = dataframe_from_series(series)
    if df.empty:
        return JSONResponse({"symbol": symbol, "agg": agg, "recommendation": None})

    # grid + best
    grid, best = compute_optimal_drop_grid(
        df,
        target_pct=target_pct,
        drop_min=drop_min,
        drop_max=drop_max,
        drop_step=drop_step,
        lookahead_days=lookahead_days,
        score_mode=score_mode,
    )

    reco = current_recommendation(
        df,
        grid=grid,
        best=best,
        target_pct=target_pct,
        peak_lookback_days=peak_lookback_days
    )

    return JSONResponse({
        "symbol": symbol,
        "agg": agg,
        "target_pct": target_pct,
        "params": {
            "drop_min": drop_min, "drop_max": drop_max, "drop_step": drop_step,
            "lookahead_days": lookahead_days, "score_mode": score_mode,
            "peak_lookback_days": peak_lookback_days
        },
        "best": best,
        "recommendation": reco
    })

@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy"
    })
