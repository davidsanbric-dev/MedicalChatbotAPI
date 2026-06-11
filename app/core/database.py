from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import DATABASE_URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    echo=False  # Set to True for SQL debugging
)

# Create session factory.
# expire_on_commit=False so instances returned from session_scope() (which commits
# on exit) keep their loaded attributes after the session closes — otherwise
# accessing them raises DetachedInstanceError.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

# Create declarative base
Base = declarative_base()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.

    Commits on success, rolls back on error, and always closes the session.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
