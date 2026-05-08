"""Tamper-evident HMAC-chained audit logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable, Optional


class AuditLogError(RuntimeError):
	"""Base error for audit log operations."""


class AuditLogIntegrityError(AuditLogError):
	"""Raised when audit chain verification fails."""


def _canonical_json(data: dict[str, Any]) -> bytes:
	return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalize_key(secret_key: str | bytes) -> bytes:
	if isinstance(secret_key, bytes):
		key = secret_key
	else:
		key = secret_key.encode("utf-8")
	if not key:
		raise AuditLogError("secret_key must not be empty")
	return key


def hash_student_id(student_id: str) -> str:
	return hashlib.sha256(student_id.encode("utf-8")).hexdigest()


def compute_entry_hmac(entry_without_hmac: dict[str, Any], secret_key: str | bytes) -> str:
	key = _normalize_key(secret_key)
	payload = _canonical_json(entry_without_hmac)
	return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _read_json_lines(log_path: Path) -> list[dict[str, Any]]:
	if not log_path.exists():
		return []
	entries: list[dict[str, Any]] = []
	with log_path.open("r", encoding="utf-8") as file:
		for line_number, line in enumerate(file, start=1):
			text = line.strip()
			if not text:
				continue
			try:
				entry = json.loads(text)
			except json.JSONDecodeError as exc:
				raise AuditLogIntegrityError(f"Invalid JSON at line {line_number}") from exc
			if not isinstance(entry, dict):
				raise AuditLogIntegrityError(f"Invalid entry type at line {line_number}")
			entries.append(entry)
	return entries


def verify_audit_chain(log_path: str | Path, secret_key: str | bytes) -> bool:
	"""Verify full HMAC chain and raise AuditLogIntegrityError on tampering."""
	key = _normalize_key(secret_key)
	path = Path(log_path)
	entries = _read_json_lines(path)

	previous_hmac = "0" * 64
	for index, entry in enumerate(entries, start=1):
		entry_hmac = entry.get("hmac")
		if not isinstance(entry_hmac, str):
			raise AuditLogIntegrityError(f"Missing or invalid hmac at entry {index}")

		signed_entry = dict(entry)
		signed_entry.pop("hmac", None)

		prev = signed_entry.get("prev_hmac")
		if prev != previous_hmac:
			raise AuditLogIntegrityError(
				f"Chain break at entry {index}: expected prev_hmac {previous_hmac}, got {prev}"
			)

		expected = hmac.new(key, _canonical_json(signed_entry), hashlib.sha256).hexdigest()
		if not hmac.compare_digest(entry_hmac, expected):
			raise AuditLogIntegrityError(f"HMAC mismatch at entry {index}")

		previous_hmac = entry_hmac

	return True


@dataclass
class AuditLogger:
	"""Append and verify tamper-evident audit events."""

	log_path: str | Path
	secret_key: str | bytes

	def __post_init__(self) -> None:
		self._path = Path(self.log_path)
		self._key = _normalize_key(self.secret_key)
		self._path.parent.mkdir(parents=True, exist_ok=True)

	def read_entries(self) -> list[dict[str, Any]]:
		return _read_json_lines(self._path)

	def append_event(
		self,
		result: str,
		student_id: Optional[str] = None,
		session_token: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		timestamp: Optional[str] = None,
		activity_type: Optional[str] = None,
	) -> dict[str, Any]:
		entries = self.read_entries()
		previous_hmac = entries[-1]["hmac"] if entries else "0" * 64

		ts = timestamp or datetime.now(timezone.utc).isoformat()
		entry: dict[str, Any] = {
			"timestamp": ts,
			"activity_type": activity_type,
			"result": result,
			"student_id": student_id,
			"student_hash": hash_student_id(student_id) if student_id else None,
			"session_token": session_token,
			"metadata": metadata or {},
			"prev_hmac": previous_hmac,
		}
		entry["hmac"] = compute_entry_hmac(entry, self._key)

		with self._path.open("a", encoding="utf-8") as file:
			file.write(json.dumps(entry, sort_keys=True) + "\n")

		return entry

	def verify(self) -> bool:
		return verify_audit_chain(self._path, self._key)


def append_event(
	log_path: str | Path,
	secret_key: str | bytes,
	result: str,
	student_id: Optional[str] = None,
	session_token: Optional[str] = None,
	metadata: Optional[dict[str, Any]] = None,
	timestamp: Optional[str] = None,
	activity_type: Optional[str] = None,
) -> dict[str, Any]:
	"""Convenience functional API for one-off event appends."""
	logger = AuditLogger(log_path=log_path, secret_key=secret_key)
	return logger.append_event(
		result=result,
		student_id=student_id,
		session_token=session_token,
		metadata=metadata,
		timestamp=timestamp,
		activity_type=activity_type,
	)


def read_entries(log_path: str | Path) -> Iterable[dict[str, Any]]:
	"""Convenience functional API for loading entries."""
	return _read_json_lines(Path(log_path))
