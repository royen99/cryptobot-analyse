# app/db.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from urllib.parse import quote_plus

from .config import get_config

cfg = get_config().database
DATABASE_URL = (
    f"postgresql+asyncpg://{cfg.user}:{quote_plus(cfg.password)}@{cfg.host}:{cfg.port}/{cfg.name}"
)

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
