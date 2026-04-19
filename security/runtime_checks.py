"""Runtime security validation helpers for production readiness."""

from __future__ import annotations

import importlib
import os
from typing import Tuple


class RuntimeSecurityError(RuntimeError):
	"""Raised when runtime security requirements are not met."""


def get_runtime_mode() -> str:
	mode = os.getenv("SFRS_MODE", "development").strip().lower()
	if mode in {"prod", "production"}:
		return "production"
	if mode in {"dev", "development", "test"}:
		return mode
	return "development"


def require_keyring_only(mode: str | None = None) -> bool:
	current_mode = (mode or get_runtime_mode()).lower()
	if os.getenv("SFRS_KEYRING_ONLY") == "1":
		return True
	return current_mode == "production"


def allow_plaintext_database_fallback(mode: str | None = None) -> bool:
	current_mode = (mode or get_runtime_mode()).lower()
	if os.getenv("SFRS_ALLOW_PLAINTEXT_DB") == "1":
		return True
	if os.getenv("SFRS_ALLOW_PLAINTEXT_DB") == "0":
		return False
	return current_mode != "production"


def _module_available(module_name: str) -> bool:
	try:
		importlib.import_module(module_name)
		return True
	except Exception:
		return False


def validate_runtime_security(mode: str | None = None) -> Tuple[bool, list[str]]:
	"""Validate runtime security posture and return (ok, issues)."""
	current_mode = (mode or get_runtime_mode()).lower()
	issues: list[str] = []

	if require_keyring_only(current_mode) and os.getenv("DB_KEY"):
		issues.append("DB_KEY environment fallback is not allowed in production/keyring-only mode")

	if not allow_plaintext_database_fallback(current_mode) and not _module_available("sqlcipher3"):
		issues.append("sqlcipher3 is required when plaintext DB fallback is disabled")

	if require_keyring_only(current_mode) and not _module_available("keyring"):
		issues.append("keyring package is required in production/keyring-only mode")

	return (len(issues) == 0), issues


def assert_runtime_security(mode: str | None = None) -> None:
	ok, issues = validate_runtime_security(mode)
	if not ok:
		raise RuntimeSecurityError("; ".join(issues))
