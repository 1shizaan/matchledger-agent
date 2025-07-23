import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool

Base = declarative_base()

def get_database_url() -> str:
    url = os.getenv("SQLALCHEMY_DATABASE_URL")
    if not url:
        raise ValueError("Database URL not configured")
    if "postgresql://" not in url:
        url = url.replace("postgres://", "postgresql://")  # SQLAlchemy 2.0 requirement
    return url

engine = create_engine(
    get_database_url(),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from app.models.user import User
    from app.models.reconciliation import Reconciliation
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()