"""Database key management using the OS credential store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
	import keyring
except ImportError:  # pragma: no cover - fallback for minimal environments
	keyring = None  # type: ignore[assignment]


class KeyManagerError(RuntimeError):
	"""Raised when database key operations fail."""


def normalize_db_key(key: str | bytes) -> str:
	if isinstance(key, bytes):
		key = key.decode("utf-8")
	key = key.strip()
	if not key:
		raise KeyManagerError("database key must not be empty")
	return key


def get_key_from_env(env_var: str = "DB_KEY") -> Optional[str]:
	value = os.getenv(env_var)
	if value is None:
		return None
	value = value.strip()
	return value or None


def set_db_key(
	key: str | bytes,
	service_name: str = "sfrs",
	username: str = "db_key",
) -> str:
	"""Persist the database key in the OS credential store."""
	normalized = normalize_db_key(key)
	if keyring is None:
		raise KeyManagerError("keyring is not installed")
	keyring.set_password(service_name, username, normalized)
	return normalized


def get_db_key(
	service_name: str = "sfrs",
	username: str = "db_key",
	env_var: str = "DB_KEY",
	allow_env_fallback: bool = True,
) -> str:
	"""Read the database key from keyring, with optional environment fallback."""
	if keyring is not None:
		stored = keyring.get_password(service_name, username)
		if stored:
			return normalize_db_key(stored)

	if allow_env_fallback:
		env_value = get_key_from_env(env_var)
		if env_value:
			return normalize_db_key(env_value)

	raise KeyManagerError(
		f"Database key not found in keyring service={service_name!r} username={username!r}"
	)


def delete_db_key(service_name: str = "sfrs", username: str = "db_key") -> None:
	"""Delete the database key from keyring if supported."""
	if keyring is None:
		raise KeyManagerError("keyring is not installed")
	try:
		keyring.delete_password(service_name, username)
	except keyring.errors.PasswordDeleteError as exc:  # type: ignore[attr-defined]
		raise KeyManagerError(str(exc)) from exc


@dataclass(frozen=True)
class DatabaseKeyConfig:
	service_name: str = "sfrs"
	username: str = "db_key"
	env_var: str = "DB_KEY"

	def load(self, allow_env_fallback: bool = True) -> str:
		return get_db_key(
			service_name=self.service_name,
			username=self.username,
			env_var=self.env_var,
			allow_env_fallback=allow_env_fallback,
		)

	def store(self, key: str | bytes) -> str:
		return set_db_key(key, service_name=self.service_name, username=self.username)
