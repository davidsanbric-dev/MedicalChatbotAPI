"""What every repository shares: the table it stores, and how it talks to the database.

Reads use a plain session, so nothing they do is ever committed. Writes use a transaction,
committed when the block ends and rolled back if it raises.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

from sqlalchemy import ColumnElement, Executable, Row, delete
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import SessionLocal
from app.core.models import Base


class BaseRepository:
    model: ClassVar[type[Base]]  # the table this repository stores

    def __init__(self, session_factory: sessionmaker[Session] = SessionLocal) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """A session for reading."""
        with self._session_factory() as session:
            yield session

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        """A session for writing: committed when the block ends, rolled back if it raises."""
        with self._session_factory.begin() as session:
            yield session

    def _scalars(self, query: Executable) -> list[Any]:
        """The first column of every row the query returns."""
        with self._session() as session:
            return list(session.scalars(query))

    def _rows(self, query: Executable) -> list[Row[Any]]:
        with self._session() as session:
            return list(session.execute(query))

    def _delete_where(self, *conditions: ColumnElement[bool]) -> int:
        """Delete this table's rows that match every condition; returns how many were removed."""
        with self._transaction() as session:
            return session.execute(delete(self.model).where(*conditions)).rowcount
