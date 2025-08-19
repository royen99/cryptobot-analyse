# app/analysis.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DropRecoveryEvent:
    start: str                 # ISO of peak time before drop
    trough: str                # ISO of deepest drawdown time
    recovery: Optional[str]    # ISO when price recovered to ~peak (or None)
    drop_pct: float            # (trough/peak - 1)
    recovery_days: Optional[float]   # (recovery - trough) in days
    overshoot_pct: Optional[float]   # max gain above old peak within overshoot window (or None)


def _to_iso(ts: pd.Timestamp | datetime | None) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(microsecond=0).isoformat()
    return ts.replace(microsecond=0).isoformat()


def compute_drawdown_events(
    df: pd.DataFrame,
    *,
    min_drop: float = 0.10,              # e.g. 0.10 = -10% drop threshold
    recovery_tolerance: float = 0.0,     # e.g. 0.005 = accept 0.5% below prior peak as "recovered"
    lookahead_days: Optional[int] = None,  # cap how long we wait for recovery from trough
    overshoot_window_days: int = 14      # measure overshoot within N days after recovery
) -> Tuple[List[DropRecoveryEvent], Dict]:
    """
    df columns required:
      - timestamp (datetime64[ns])
      - price (float)

    Returns (events, summary).
    """
    if df.empty:
        return [], {
            "avg_recovery_days": None,
            "median_recovery_days": None,
            "success_rate": None,
            "worst_recovery_days": None,
            "events": 0
        }

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values
    px = df["price"].astype(float).values

    events: List[DropRecoveryEvent] = []

    peak_price = px[0]
    peak_time = pd.to_datetime(ts[0])
    in_drawdown = False
    trough_price = peak_price
    trough_time = peak_time

    # Helper: find overshoot after recovery
    def overshoot_after(recovery_idx: int, old_peak: float, recovery_time: pd.Timestamp) -> Optional[float]:
        if overshoot_window_days <= 0:
            return None
        end_time = recovery_time + pd.Timedelta(days=overshoot_window_days)
        # slice indices
        i = recovery_idx
        while i < len(ts) and pd.to_datetime(ts[i]) <= end_time:
            i += 1
        if i <= recovery_idx:
            return None
        max_after = float(np.max(px[recovery_idx:i]))
        return (max_after / old_peak) - 1.0

    # Main scan
    for i in range(1, len(px)):
        cur_price = px[i]
        cur_time = pd.to_datetime(ts[i])

        # new peak resets state if not in drawdown
        if not in_drawdown and cur_price > peak_price:
            peak_price = cur_price
            peak_time = cur_time
            continue

        # drawdown from current peak
        dd = (cur_price / peak_price) - 1.0

        if not in_drawdown:
            # Enter drawdown if threshold breached
            if dd <= -min_drop:
                in_drawdown = True
                trough_price = cur_price
                trough_time = cur_time
            # else just continue (still within minor pullback)
        else:
            # Update trough while in drawdown
            if cur_price < trough_price:
                trough_price = cur_price
                trough_time = cur_time

            # Optional: enforce lookahead limit from trough
            if lookahead_days is not None:
                if (cur_time - trough_time) > pd.Timedelta(days=lookahead_days):
                    # Give up on recovery, record unrecovered event
                    events.append(
                        DropRecoveryEvent(
                            start=_to_iso(peak_time),
                            trough=_to_iso(trough_time),
                            recovery=None,
                            drop_pct=(trough_price / peak_price) - 1.0,
                            recovery_days=None,
                            overshoot_pct=None,
                        )
                    )
                    # After an unrecovered long drawdown, restart peak from here
                    peak_price = cur_price
                    peak_time = cur_time
                    in_drawdown = False
                    continue

            # Recovered? price back to ~peak (within tolerance)
            recovered_level = peak_price * (1.0 - recovery_tolerance)
            if cur_price >= recovered_level:
                # record event
                drop_pct = (trough_price / peak_price) - 1.0
                rec_days = (cur_time - trough_time).total_seconds() / 86400.0
                over = overshoot_after(i, peak_price, cur_time)
                events.append(
                    DropRecoveryEvent(
                        start=_to_iso(peak_time),
                        trough=_to_iso(trough_time),
                        recovery=_to_iso(cur_time),
                        drop_pct=drop_pct,
                        recovery_days=rec_days,
                        overshoot_pct=over,
                    )
                )
                # reset: new peak could be current price OR keep rolling peak logic
                peak_price = max(cur_price, peak_price)
                peak_time = cur_time if cur_price >= peak_price else peak_time
                in_drawdown = False
                trough_price = peak_price
                trough_time = peak_time
            # else remain in drawdown

    # If we end while in drawdown, record partial (unrecovered)
    if in_drawdown:
        events.append(
            DropRecoveryEvent(
                start=_to_iso(peak_time),
                trough=_to_iso(trough_time),
                recovery=None,
                drop_pct=(trough_price / peak_price) - 1.0,
                recovery_days=None,
                overshoot_pct=None,
            )
        )

    # Summary stats
    rec_days_list = [e.recovery_days for e in events if e.recovery_days is not None]
    summary = {
        "avg_recovery_days": float(np.mean(rec_days_list)) if rec_days_list else None,
        "median_recovery_days": float(np.median(rec_days_list)) if rec_days_list else None,
        "success_rate": float(len(rec_days_list) / len(events)) if events else None,
        "worst_recovery_days": float(np.max(rec_days_list)) if rec_days_list else None,
        "events": len(events),
    }

    return events, summary


def dataframe_from_series(series: List[Tuple[str, float]]) -> pd.DataFrame:
    """
    series: [(timestamp_iso, price_float), ...]
    Ensures proper dtypes and sorting.
    """
    df = pd.DataFrame(series, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@dataclass
class DropGridPoint:
    drop_pct: float                # threshold (e.g., 0.12 for 12%)
    entries: int                   # number of historical entries
    successes: int                 # how many hit target within lookahead
    success_rate: Optional[float]  # successes / entries
    median_days: Optional[float]   # median time-to-target on successes
    mean_days: Optional[float]     # mean time-to-target on successes
    p90_days: Optional[float]      # 90th percentile time-to-target on successes
    score: Optional[float]         # chooser metric (success_rate / median_days)

def _simulate_drop_threshold(
    df: pd.DataFrame,
    drop_pct: float,
    target_pct: float,
    lookahead_days: int
) -> Tuple[int, int, List[float], float]:
    """
    Returns:
      entries, successes, times_to_target_days (successes only), total_time_in_position_days
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values
    px = df["price"].astype(float).values

    entries = 0
    successes = 0
    ttt_days: List[float] = []
    time_in_position_days = 0.0

    peak = px[0]
    entry_idx: Optional[int] = None
    entry_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None

    for i in range(1, len(px)):
        cur = px[i]
        cur_t = pd.to_datetime(ts[i])

        if entry_idx is None and cur > peak:
            peak = cur

        if entry_idx is None:
            if cur <= peak * (1.0 - drop_pct):
                entry_idx = i
                entry_time = cur_t
                entry_price = cur
                entries += 1
        else:
            assert entry_price is not None and entry_time is not None
            if cur >= entry_price * (1.0 + target_pct):
                dt = (cur_t - entry_time).total_seconds() / 86400.0
                successes += 1
                ttt_days.append(dt)
                time_in_position_days += dt
                peak = cur
                entry_idx = None
                entry_time = None
                entry_price = None
                continue

            if (cur_t - entry_time) > pd.Timedelta(days=lookahead_days):
                # timeout (failure)
                time_in_position_days += float(lookahead_days)
                peak = cur
                entry_idx = None
                entry_time = None
                entry_price = None
                continue

    # no penalty for an open trade at the very end (can add if you want)
    return entries, successes, ttt_days, time_in_position_days

def compute_optimal_drop_grid(
    df: pd.DataFrame,
    *,
    target_pct: float = 0.05,
    drop_min: float = 0.05,
    drop_max: float = 0.40,
    drop_step: float = 0.01,
    lookahead_days: int = 60,
    score_mode: str = "expected_return_per_year",  # or: success_rate | successes_per_year | sr_over_median_days
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Adds frequency-aware fields:
      - entries_per_year, successes_per_year, expected_return_per_year (= successes_per_year * target_pct)
      - occupancy (fraction of time in trade), approx via total time-in-position
    'score_mode':
      - "success_rate":              maximize success rate
      - "successes_per_year":        maximize wins per year
      - "expected_return_per_year":  maximize wins/year * target_pct
      - "sr_over_median_days":       maximize (success_rate / median_days)
    """
    if df.empty:
        return [], None

    df = df.sort_values("timestamp").reset_index(drop=True)
    total_days = max(1.0, (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days)
    total_years = total_days / 365.25

    drops = np.arange(drop_min, drop_max + 1e-9, drop_step)
    grid: List[Dict] = []
    best: Optional[Dict] = None

    def score_of(row: Dict) -> float:
        if score_mode == "success_rate":
            return float(row["success_rate"] or 0.0)
        if score_mode == "successes_per_year":
            return float(row["successes_per_year"] or 0.0)
        if score_mode == "expected_return_per_year":
            return float(row["expected_return_per_year"] or 0.0)
        if score_mode == "sr_over_median_days":
            sr = row["success_rate"]
            md = row["median_days"]
            return float((sr or 0.0) / md) if (sr and md and md > 0) else 0.0
        return float(row.get("score", 0.0))

    for d in drops:
        entries, successes, ttt, time_pos_days = _simulate_drop_threshold(df, d, target_pct, lookahead_days)
        success_rate = (successes / entries) if entries > 0 else None
        med = float(np.median(ttt)) if ttt else None
        mean = float(np.mean(ttt)) if ttt else None
        p90 = float(np.percentile(ttt, 90)) if ttt else None

        entries_per_year = (entries / total_years) if total_years > 0 else None
        successes_per_year = (successes / total_years) if total_years > 0 else None
        expected_return_per_year = (successes_per_year * target_pct) if successes_per_year is not None else None
        occupancy = time_pos_days / total_days  # fraction of time in trade (0..1)

        row = {
            "drop_pct": float(d),
            "entries": int(entries),
            "successes": int(successes),
            "success_rate": float(success_rate) if success_rate is not None else None,
            "median_days": med,
            "mean_days": mean,
            "p90_days": p90,
            "entries_per_year": float(entries_per_year) if entries_per_year is not None else None,
            "successes_per_year": float(successes_per_year) if successes_per_year is not None else None,
            "expected_return_per_year": float(expected_return_per_year) if expected_return_per_year is not None else None,
            "occupancy": float(occupancy),
        }
        row["score"] = score_of(row)
        grid.append(row)

        if best is None or row["score"] > (best["score"] or 0.0):
            best = row

    if best is not None:
        best = dict(best)
        best["score_mode"] = score_mode

    return grid, best

def _entry_times_for_drop(df: pd.DataFrame, drop_pct: float) -> List[pd.Timestamp]:
    """
    Returns timestamps when a 'buy signal' would have been generated:
    price falls >= drop_pct from a rolling peak (and we're not already 'in trade').
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values
    px = df["price"].astype(float).values

    peak = px[0]
    in_trade = False
    entries: List[pd.Timestamp] = []

    for i in range(1, len(px)):
        cur = px[i]
        cur_t = pd.to_datetime(ts[i])

        if not in_trade and cur > peak:
            peak = cur

        if not in_trade:
            if cur <= peak * (1.0 - drop_pct):
                # signal
                entries.append(cur_t)
                in_trade = True
        else:
            # exit "trade" when we make a new peak (simple reset)
            if cur > peak:
                peak = cur
                in_trade = False

    return entries

def _median_inter_signal_days(entry_times: List[pd.Timestamp]) -> Optional[float]:
    """
    Robustly compute median gap (in days) between successive entry times.
    Accepts a list of pandas/py datetimes and handles empty/1-item lists.
    """
    # Normalize to numpy datetime64[ns] and drop nulls
    clean = [pd.to_datetime(t) for t in entry_times if pd.notna(t)]
    if len(clean) < 2:
        return None
    arr = np.array(clean, dtype="datetime64[ns]")
    arr.sort()
    # np.diff now yields timedelta64[ns]; convert to days as float
    diffs_days = np.diff(arr).astype("timedelta64[s]").astype("float64") / (24 * 3600.0)
    if diffs_days.size == 0:
        return None
    return float(np.median(diffs_days))

def current_recommendation(
    df: pd.DataFrame,
    *,
    grid: List[Dict],
    best: Optional[Dict],
    target_pct: float,
    peak_lookback_days: Optional[int] = 180
) -> Dict:
    """
    Uses the 'best' drop threshold from the optimal grid and the latest price to
    say: BUY NOW or WAIT, plus rough timing estimates.
    """
    out = {
        "action": None,  # "buy" | "wait" | None
        "as_of": None,
        "current_price": None,
        "peak_price": None,
        "current_drawdown_pct": None,
        "recommended_drop_pct": None,
        "target_buy_price": None,
        "distance_to_signal_pct": None,
        "expected_wait_days": None,
        "expected_median_days_to_target": None,
        "success_rate": None,
        "entries_per_year": None,
        "successes_per_year": None,
        "expected_return_per_year": None,
    }
    if df.empty or not best:
        return out

    df = df.sort_values("timestamp").reset_index(drop=True)
    last_row = df.iloc[-1]
    now_ts = pd.to_datetime(last_row["timestamp"])
    cur_price = float(last_row["price"])

    # Peak window (default 180d) to anchor "recent peak"
    if peak_lookback_days and peak_lookback_days > 0:
        start_win = now_ts - pd.Timedelta(days=peak_lookback_days)
        df_win = df[df["timestamp"] >= start_win]
        if df_win.empty:
            df_win = df
    else:
        df_win = df

    peak_price = float(df_win["price"].max())
    dd_cur = (peak_price - cur_price) / peak_price if peak_price > 0 else 0.0  # fraction in [0,1]

    # Best settings from grid
    d = float(best["drop_pct"])
    out.update({
        "as_of": now_ts.replace(microsecond=0).isoformat(),
        "current_price": cur_price,
        "peak_price": peak_price,
        "current_drawdown_pct": dd_cur,
        "recommended_drop_pct": d,
        "success_rate": best.get("success_rate"),
        "entries_per_year": best.get("entries_per_year"),
        "successes_per_year": best.get("successes_per_year"),
        "expected_return_per_year": best.get("expected_return_per_year"),
        "expected_median_days_to_target": best.get("median_days"),
    })

    target_buy_price = peak_price * (1.0 - d)
    out["target_buy_price"] = target_buy_price

    if dd_cur >= d - 1e-9:
        # Already at or beyond the threshold → buy now
        out["action"] = "buy"
        out["distance_to_signal_pct"] = 0.0
        out["expected_wait_days"] = 0.0
        return out

    # Otherwise, WAIT until price declines to target_buy_price
    out["action"] = "wait"
    out["distance_to_signal_pct"] = d - dd_cur

    # Estimate expected wait:
    entries = _entry_times_for_drop(df, d)
    median_gap = _median_inter_signal_days(entries)

    elapsed = None
    if entries:
        elapsed = (now_ts - pd.to_datetime(entries[-1])).total_seconds() / 86400.0

    # Heuristic for expected wait
    if median_gap is not None and elapsed is not None:
        exp_wait = max(0.0, float(median_gap - elapsed))
    else:
        epy = best.get("entries_per_year")
        exp_wait = float(365.25 / epy) if epy and epy > 0 else None

    out["expected_wait_days"] = exp_wait
    return out
