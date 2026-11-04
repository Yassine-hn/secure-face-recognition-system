"""Centralized service initialization for desktop, web, and CLI environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.enrollment import EnrollmentService
from core.matcher import FaceMatcher
from core.recognition import RecognitionService
from liveness.active import ActiveLivenessOrchestrator
from liveness.passive import PassiveLivenessChecker
from security.audit_log import AuditLogger
from security.cancelable import CancelableTransformer
from security.database import EnrollmentDatabase
from security.key_manager import get_db_key
from security.runtime_checks import (
	allow_plaintext_database_fallback,
	assert_runtime_security,
	get_runtime_mode,
	require_keyring_only,
)


@dataclass
class AppServices:
	detector: Any
	embedder: Any
	matcher: FaceMatcher
	recognition: RecognitionService
	passive_liveness: PassiveLivenessChecker
	active_liveness: ActiveLivenessOrchestrator
	enrollment: EnrollmentService
	audit_logger: AuditLogger
	database: EnrollmentDatabase
	transformer: CancelableTransformer


def build_default_services(base_dir: str | Path | None = None) -> AppServices:
	"""Build and wire all security, biometric, and audit services."""
	mode = get_runtime_mode()
	assert_runtime_security(mode)
	keyring_only = require_keyring_only(mode)
	allow_plaintext = allow_plaintext_database_fallback(mode)

	base_path = Path(base_dir or Path(__file__).resolve().parents[1])
	data_dir = base_path / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	db_path = data_dir / "embeddings.db"
	audit_path = data_dir / "audit.log"
	db_key = get_db_key(allow_env_fallback=not keyring_only)

	detector = FaceDetector()
	embedder = FaceEmbedder()
	# Share lazy InsightFace instance so models are only initialized once
	original_detector_load = detector._load_model

	def _shared_load():
		m = original_detector_load()
		embedder._model = m
		return m

	detector._load_model = _shared_load
	embedder._load_model = _shared_load
	matcher = FaceMatcher(threshold=0.5)
	# Use the same transformer for both enrollment and recognition
	# This ensures query embeddings are transformed the same way as stored templates
	transformer = CancelableTransformer(seed=42)
	recognition = RecognitionService(
		detector=detector,
		embedder=embedder,
		matcher=matcher,
		transformer=transformer,
	)
	passive_liveness = PassiveLivenessChecker()
	active_liveness = ActiveLivenessOrchestrator()
	database = EnrollmentDatabase(
		db_path=db_path,
		db_key=db_key,
		allow_plaintext_fallback=allow_plaintext,
	)
	enrollment = EnrollmentService(database=database, transformer=transformer)
	audit_logger = AuditLogger(log_path=audit_path, secret_key=db_key)

	return AppServices(
		detector=detector,
		embedder=embedder,
		matcher=matcher,
		recognition=recognition,
		passive_liveness=passive_liveness,
		active_liveness=active_liveness,
		enrollment=enrollment,
		audit_logger=audit_logger,
		database=database,
		transformer=transformer,
	)
