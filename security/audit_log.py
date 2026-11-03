"""Tamper-evident HMAC-chained audit logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Optional


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


def _read_json_lines_from_end(log_path: Path, count: int) -> list[dict[str, Any]]:
	if count <= 0 or not log_path.exists():
		return []

	block_size = 8192
	collected: list[bytes] = []
	buffer = b""
	with log_path.open("rb") as file:
		file.seek(0, 2)
		position = file.tell()
		while position > 0 and len(collected) < count:
			read_size = min(block_size, position)
			position -= read_size
			file.seek(position)
			chunk = file.read(read_size)
			buffer = chunk + buffer
			parts = buffer.splitlines()
			if position > 0:
				buffer = parts[0] if parts else b""
				complete_lines = parts[1:]
			else:
				buffer = b""
				complete_lines = parts

			for line in reversed(complete_lines):
				if not line.strip():
					continue
				collected.insert(0, line)
				if len(collected) >= count:
					break

		if position == 0 and buffer.strip() and len(collected) < count:
			collected.insert(0, buffer)

	parsed: list[dict[str, Any]] = []
	for line in collected:
		try:
			entry = json.loads(line.decode("utf-8"))
		except json.JSONDecodeError as exc:
			raise AuditLogIntegrityError("Invalid JSON while reading recent audit entries") from exc
		if not isinstance(entry, dict):
			raise AuditLogIntegrityError("Invalid entry type while reading recent audit entries")
		parsed.append(entry)
	return parsed


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


# Alias for backward compatibility
verify_chain_integrity = verify_audit_chain


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

	def read_entries_page(self, page_size: int = 10, page_index: int = 0) -> tuple[list[dict[str, Any]], bool]:
		return read_entries_page(self._path, page_size=page_size, page_index=page_index)

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

	def log_event(
		self,
		activity_type: Optional[str] = None,
		result: str = "GRANTED",
		student_id: Optional[str] = None,
		session_token: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		timestamp: Optional[str] = None,
	) -> dict[str, Any]:
		"""Convenience alias for append_event with activity_type first."""
		return self.append_event(
			result=result,
			student_id=student_id,
			session_token=session_token,
			metadata=metadata,
			timestamp=timestamp,
			activity_type=activity_type,
		)


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


def read_entries(log_path: str | Path) -> list[dict[str, Any]]:
	"""Convenience functional API for loading all entries."""
	return _read_json_lines(Path(log_path))


def read_entries_page(
	log_path: str | Path,
	page_size: int = 10,
	page_index: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
	"""Load one page of entries from the end of the log.

	Returns (entries, has_next_page). Page 0 is the newest page.
	"""
	if page_size <= 0 or page_index < 0:
		return [], False

	path = Path(log_path)
	offset = page_index * page_size
	window = _read_json_lines_from_end(path, offset + page_size + 1)
	if len(window) <= offset:
		return [], False

	page_window = window[:-offset] if offset else window
	has_next = len(page_window) > page_size
	if has_next:
		page_rows = page_window[1 : page_size + 1]
	else:
		page_rows = page_window[:page_size]
	return page_rows, has_next
