"""Enrollment helpers for collecting embeddings and persisting templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from security.cancelable import CancelableTransformer
from security.database import EnrollmentDatabase, EnrollmentRecord


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
	return _call_first_supported(detector, ["detect_face", "detect", "extract_face"], frame)


def _extract_embedding(embedder: Any, face: Any) -> np.ndarray:
	embedding = _call_first_supported(embedder, ["embed", "extract_embedding", "encode"], face)
	return _as_embedding(embedding)


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
	while len(embeddings) < sample_count and attempts < max_attempts:
		attempts += 1
		frame = frame_source()
		if frame is None:
			continue
		face = _detect_face(face_detector, frame)
		if face is None:
			continue
		try:
			embeddings.append(_extract_embedding(embedder, face))
		except EnrollmentError:
			continue

	if len(embeddings) < sample_count:
		raise EnrollmentError(
			f"Only collected {len(embeddings)} embeddings after {attempts} attempts"
		)
	return embeddings


@dataclass
class EnrollmentService:
	database: EnrollmentDatabase
	transformer: CancelableTransformer | None = None

	def enroll_embedding(
		self,
		student_id: str,
		embedding: np.ndarray | list[float] | tuple[float, ...],
		metadata: Optional[dict[str, Any]] = None,
	) -> EnrollmentRecord:
		average = _as_embedding(embedding)
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

