from decimal import Decimal
from typing import Iterable, List, Tuple

def to_float_rows(rows: Iterable[Tuple]) -> List[Tuple]:
    out = []
    for ts, price in rows:
        # price is Decimal due to Numeric(18,12)
        out.append((ts.isoformat(), float(price) if price is not None else None))
    return out
