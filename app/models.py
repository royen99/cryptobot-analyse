from sqlalchemy import Column, Text, Numeric, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class PriceHistory(Base):
    __tablename__ = "price_history"
    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    timestamp: Mapped = mapped_column(TIMESTAMP(timezone=False), primary_key=True)
    price: Mapped = mapped_column(Numeric(18, 12), nullable=True)
