from typing import Literal, Tuple

Agg = Literal["raw","1min","5min","15min","1h","4h","1d"]

def agg_to_trunc(agg: str) -> Tuple[str, str]:
    # returns (date_trunc unit, label)
    table = {
        "raw": ("", "raw"),
        "1min": ("minute", "1min"),
        "5min": ("minute", "5min"),
        "15min": ("minute", "15min"),
        "1h": ("hour", "1h"),
        "4h": ("hour", "4h"),
        "1d": ("day", "1d"),
    }
    return table.get(agg or "15min", ("minute", "15min"))

def minute_bucket_sql(agg: str) -> str:
    """
    Returns a SQL expression that buckets "timestamp" for the requested agg.
    Avoids make_interval() for compatibility.
    """
    if agg == "raw":
        return ""
    if agg == "1min":
        return 'date_trunc(\'minute\', "timestamp")'
    if agg == "5min":
        # floor to previous 5-min boundary
        return (
            'date_trunc(\'minute\', "timestamp")'
            ' - ((extract(minute from "timestamp")::int % 5) * interval \'1 minute\')'
        )
    if agg == "15min":
        # floor to previous 15-min boundary
        return (
            'date_trunc(\'minute\', "timestamp")'
            ' - ((extract(minute from "timestamp")::int % 15) * interval \'1 minute\')'
        )
    if agg == "1h":
        return 'date_trunc(\'hour\', "timestamp")'
    if agg == "4h":
        # floor to previous 4-hour boundary
        return (
            'date_trunc(\'hour\', "timestamp")'
            ' - ((extract(hour from "timestamp")::int % 4) * interval \'1 hour\')'
        )
    if agg == "1d":
        return 'date_trunc(\'day\', "timestamp")'
    # sensible default
    return 'date_trunc(\'minute\', "timestamp")'
