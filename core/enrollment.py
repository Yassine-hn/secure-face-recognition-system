"""Enrollment helpers for collecting embeddings and persisting templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from core.matcher import cosine_similarity
from security.cancelable import CancelableTransformer
from security.database import EnrollmentDatabase, EnrollmentRecord

DUPLICATE_THRESHOLD = 0.85


class EnrollmentError(RuntimeError):
	"""Raised when enrollment fails."""


def _as_embedding(value: Any) -> np.ndarray:
	embedding = np.asarray(value, dtype=np.float32)
	if embedding.ndim != 1:
		raise EnrollmentError(f"embedding must be 1-D, got shape {embedding.shape}")
	if embedding.size == 0:
		raise EnrollmentError("embedding must not be empty")
	return embedding


def average_embeddings(embeddings: Iterable[np.ndarray | list[float] | tuple[float, ...]]) -> np.ndarray:
	vectors = [_as_embedding(embedding) for embedding in embeddings]
	if not vectors:
		raise EnrollmentError("at least one embedding is required")
	first_shape = vectors[0].shape
	if any(vector.shape != first_shape for vector in vectors[1:]):
		raise EnrollmentError("all embeddings must have the same shape")
	return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def _call_first_supported(target: Any, method_names: list[str], *args: Any) -> Any:
	if callable(target):
		return target(*args)
	for method_name in method_names:
		method = getattr(target, method_name, None)
		if callable(method):
			return method(*args)
	raise EnrollmentError(f"Unsupported callable target: {target!r}")


def _detect_face(detector: Any, frame: Any) -> Any:
	for method_name in ["detect", "detect_face", "extract_face"]:
		method = getattr(detector, method_name, None)
		if callable(method):
			return method(frame)
	if callable(detector):
		return detector(frame)
	raise EnrollmentError(f"Unsupported detector target: {detector!r}")


def _extract_embedding(embedder: Any, face: Any) -> np.ndarray:
	embedding = _call_first_supported(embedder, ["embed", "extract_embedding", "encode"], face)
	return _as_embedding(embedding)


def _extract_embedding_from_frame(embedder: Any, frame: Any) -> np.ndarray:
	"""Extract embedding using full-frame processing if available."""
	if hasattr(embedder, "extract_embedding_from_frame"):
		embedding, _ = embedder.extract_embedding_from_frame(frame)
		return _as_embedding(embedding)
	raise EnrollmentError("Embedder does not support full-frame extraction")


def collect_embeddings(
	frame_source: Callable[[], Any],
	face_detector: Any,
	embedder: Any,
	sample_count: int = 5,
	max_attempts: int = 50,
) -> list[np.ndarray]:
	"""Collect embeddings from a frame source using detector and embedder dependencies."""
	if sample_count <= 0:
		raise EnrollmentError("sample_count must be positive")
	if max_attempts < sample_count:
		raise EnrollmentError("max_attempts must be >= sample_count")

	embeddings: list[np.ndarray] = []
	attempts = 0
	last_error: Exception | None = None
	while len(embeddings) < sample_count and attempts < max_attempts:
		attempts += 1
		frame = frame_source()
		if frame is None:
			continue

		# First try: full-frame extraction with ArcFace detection
		try:
			embedding = _extract_embedding_from_frame(embedder, frame)
			embeddings.append(embedding)
			continue
		except EnrollmentError:
			pass  # Fall back to detect-then-embed

		# Second try: legacy detect-then-embed pipeline
		detection = _detect_face(face_detector, frame)
		if detection is None:
			continue
		direct_embedding = getattr(detection, "embedding", None)
		if direct_embedding is not None:
			try:
				embeddings.append(_as_embedding(direct_embedding))
				continue
			except EnrollmentError as exc:
				last_error = exc
				continue
		face = getattr(detection, "face", detection)
		try:
			embeddings.append(_extract_embedding(embedder, face))
		except EnrollmentError as exc:
			last_error = exc
			continue

	if len(embeddings) < sample_count:
		error = EnrollmentError(f"Only collected {len(embeddings)} embeddings after {attempts} attempts")
		if last_error is not None:
			raise error from last_error
		raise error
	return embeddings


@dataclass
class EnrollmentService:
	database: EnrollmentDatabase
	transformer: CancelableTransformer | None = None

	@staticmethod
	def _validate_student_id(student_id: str) -> str:
		value = student_id.strip()
		if not value:
			raise EnrollmentError("student_id must not be empty")
		return value

	def _check_duplicate_face(
		self,
		embedding: np.ndarray,
		exclude_student_id: Optional[str] = None,
	) -> Optional[str]:
		"""Check if embedding matches an existing enrolled face.

		Returns the student_id of the duplicate if found, None otherwise.
		"""
		existing_records = self.database.list_enrollments()
		if not existing_records:
			return None

		query = _as_embedding(embedding)
		if self.transformer:
			query = self.transformer.transform(query)

		for record in existing_records:
			if exclude_student_id and record.student_id == exclude_student_id:
				continue
			try:
				similarity = cosine_similarity(query, record.template)
				if similarity >= DUPLICATE_THRESHOLD:
					return record.student_id
			except Exception:
				continue
		return None

	def enroll_embedding(
		self,
		student_id: str,
		embedding: np.ndarray | list[float] | tuple[float, ...],
		metadata: Optional[dict[str, Any]] = None,
	) -> EnrollmentRecord:
		student_id = self._validate_student_id(student_id)
		average = _as_embedding(embedding)
		duplicate_id = self._check_duplicate_face(average, exclude_student_id=student_id)
		if duplicate_id is not None:
			raise EnrollmentError(f"Face already enrolled as student {duplicate_id}")
		transformed = self.transformer.transform(average) if self.transformer else average
		return self.database.upsert_enrollment(student_id, transformed, metadata=metadata)

	def enroll_from_capture(
		self,
		student_id: str,
		frame_source: Callable[[], Any],
		face_detector: Any,
		embedder: Any,
		sample_count: int = 5,
		max_attempts: int = 50,
		metadata: Optional[dict[str, Any]] = None,
	) -> EnrollmentRecord:
		student_id = self._validate_student_id(student_id)
		embeddings = collect_embeddings(
			frame_source=frame_source,
			face_detector=face_detector,
			embedder=embedder,
			sample_count=sample_count,
			max_attempts=max_attempts,
		)
		averaged = average_embeddings(embeddings)
		return self.enroll_embedding(student_id, averaged, metadata=metadata)

	def rotate_templates(self, seed: int | None = None) -> np.ndarray | None:
		if self.transformer is None:
			return None
		return self.transformer.rotate(seed=seed)

