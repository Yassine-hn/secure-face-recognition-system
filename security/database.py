"""Encrypted enrollment database with SQLCipher-first, sqlite fallback support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Optional

import numpy as np

from security.key_manager import get_db_key

try:  # pragma: no cover - optional dependency
	import sqlcipher3 as sqlcipher
except ImportError:  # pragma: no cover - fallback for development/testing
	sqlcipher = None  # type: ignore[assignment]


class DatabaseError(RuntimeError):
	"""Raised when enrollment database operations fail."""


@dataclass(frozen=True)
class EnrollmentRecord:
	student_id: str
	template: np.ndarray
	metadata: dict[str, Any]
	created_at: str
	updated_at: str


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _to_blob(template: np.ndarray | Iterable[float]) -> tuple[bytes, int]:
	array = np.asarray(template, dtype=np.float32)
	if array.ndim != 1:
		raise DatabaseError(f"template must be 1-D, got shape {array.shape}")
	return array.tobytes(), int(array.shape[0])


def _from_blob(blob: bytes, dimension: int) -> np.ndarray:
	array = np.frombuffer(blob, dtype=np.float32)
	if array.size != dimension:
		raise DatabaseError(
			f"Stored template dimension mismatch: expected {dimension}, got {array.size}"
		)
	return array.copy()


class EnrollmentDatabase:
	"""SQLite/SQLCipher backed storage for transformed face templates."""

	def __init__(
		self,
		db_path: str | Path,
		db_key: str | None = None,
		allow_plaintext_fallback: bool = True,
	) -> None:
		self.path = Path(db_path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self.db_key = db_key or get_db_key()
		self.allow_plaintext_fallback = allow_plaintext_fallback
		self._lock = threading.RLock()
		self._connection = self._open_connection()
		self._initialize_schema()

	def _open_connection(self) -> sqlite3.Connection:
		if sqlcipher is not None:
			connection = sqlcipher.connect(str(self.path), check_same_thread=False)
			escaped_key = self.db_key.replace("'", "''")
			connection.execute(f"PRAGMA key = '{escaped_key}'")
			connection.execute("PRAGMA cipher_page_size = 4096")
			connection.execute("PRAGMA foreign_keys = ON")
			return connection

		if not self.allow_plaintext_fallback:
			raise DatabaseError("sqlcipher3 is not installed and plaintext fallback is disabled")

		connection = sqlite3.connect(str(self.path), check_same_thread=False)
		connection.execute("PRAGMA foreign_keys = ON")
		return connection

	def _initialize_schema(self) -> None:
		with self._lock:
			self._connection.execute(
				"""
				CREATE TABLE IF NOT EXISTS enrollments (
					student_id TEXT PRIMARY KEY,
					template BLOB NOT NULL,
					dimension INTEGER NOT NULL,
					metadata TEXT NOT NULL DEFAULT '{}',
					created_at TEXT NOT NULL,
					updated_at TEXT NOT NULL
				)
				"""
			)
			self._connection.commit()

	def close(self) -> None:
		with self._lock:
			try:
				self._connection.close()
			except Exception:
				pass

	def __enter__(self) -> "EnrollmentDatabase":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def upsert_enrollment(
		self,
		student_id: str,
		template: np.ndarray | Iterable[float],
		metadata: Optional[dict[str, Any]] = None,
	) -> EnrollmentRecord:
		student_id = student_id.strip()
		if not student_id:
			raise DatabaseError("student_id must not be empty")
		blob, dimension = _to_blob(template)
		timestamp = _now_iso()
		payload = json.dumps(metadata or {}, sort_keys=True)
		with self._lock:
			self._connection.execute(
				"""
				INSERT INTO enrollments (student_id, template, dimension, metadata, created_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?)
				ON CONFLICT(student_id) DO UPDATE SET
					template = excluded.template,
					dimension = excluded.dimension,
					metadata = excluded.metadata,
					updated_at = excluded.updated_at
				""",
				(student_id, blob, dimension, payload, timestamp, timestamp),
			)
			self._connection.commit()
		return self.get_enrollment(student_id)

	def get_enrollment(self, student_id: str) -> EnrollmentRecord:
		student_id = student_id.strip()
		if not student_id:
			raise DatabaseError("student_id must not be empty")
		with self._lock:
			cursor = self._connection.execute(
				"""
				SELECT student_id, template, dimension, metadata, created_at, updated_at
				FROM enrollments
				WHERE student_id = ?
				""",
				(student_id,),
			)
			row = cursor.fetchone()
		if row is None:
			raise DatabaseError(f"Enrollment not found for student_id={student_id!r}")
		metadata = json.loads(row[3] or "{}")
		template = _from_blob(row[1], int(row[2]))
		return EnrollmentRecord(
			student_id=row[0],
			template=template,
			metadata=metadata,
			created_at=row[4],
			updated_at=row[5],
		)

	def list_enrollments(self) -> list[EnrollmentRecord]:
		with self._lock:
			cursor = self._connection.execute(
				"""
				SELECT student_id, template, dimension, metadata, created_at, updated_at
				FROM enrollments
				ORDER BY student_id
				"""
			)
			rows = cursor.fetchall()
		records: list[EnrollmentRecord] = []
		for row in rows:
			records.append(
				EnrollmentRecord(
					student_id=row[0],
					template=_from_blob(row[1], int(row[2])),
					metadata=json.loads(row[3] or "{}"),
					created_at=row[4],
					updated_at=row[5],
				)
			)
		return records

	def delete_enrollment(self, student_id: str) -> None:
		student_id = student_id.strip()
		if not student_id:
			raise DatabaseError("student_id must not be empty")
		with self._lock:
			self._connection.execute("DELETE FROM enrollments WHERE student_id = ?", (student_id,))
			self._connection.commit()

	def clear(self) -> None:
		with self._lock:
			self._connection.execute("DELETE FROM enrollments")
			self._connection.commit()

	# Aliases for convenience
	list_records = list_enrollments
	delete_record = delete_enrollment


def create_database(db_path: str | Path, db_key: str | None = None) -> EnrollmentDatabase:
	return EnrollmentDatabase(db_path=db_path, db_key=db_key)
