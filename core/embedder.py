"""ArcFace embedding extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


class EmbeddingError(RuntimeError):
	"""Raised when embedding extraction fails or invalid inputs are supplied."""


def _as_face_image(face: Any) -> np.ndarray:
	image = np.asarray(face)
	if image.ndim == 1:
		if image.size == 0:
			raise EmbeddingError("face vector must not be empty")
		return image.astype(np.float32)
	if image.ndim not in (2, 3):
		raise EmbeddingError(f"face must be a 1-D embedding or 2-D/3-D image, got shape {image.shape}")
	if image.size == 0:
		raise EmbeddingError("face image must not be empty")
	return image


def normalize_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
	vector = np.asarray(embedding, dtype=np.float32)
	if vector.ndim != 1:
		raise EmbeddingError(f"embedding must be a 1-D vector, got shape {vector.shape}")
	if vector.size == 0:
		raise EmbeddingError("embedding must not be empty")
	norm = float(np.linalg.norm(vector))
	if norm == 0.0:
		raise EmbeddingError("embedding norm must not be zero")
	return (vector / norm).astype(np.float32)


@dataclass(frozen=True)
class EmbeddingResult:
	vector: np.ndarray
	face_count: int = 1


class FaceEmbedder:
	"""Extract a normalized 512-D ArcFace embedding.

	The class uses InsightFace when available and also accepts injected models or
	precomputed embeddings for tests and lightweight environments.
	"""

	def __init__(
		self,
		model: Any | None = None,
		model_name: str = "buffalo_l",
		providers: Optional[list[str]] = None,
		ctx_id: int = -1,
		det_size: tuple[int, int] = (640, 640),
	) -> None:
		self._model = model
		self.model_name = model_name
		self.providers = providers
		self.ctx_id = ctx_id
		self.det_size = det_size

	def _load_model(self) -> Any:
		if self._model is not None:
			return self._model

		try:
			from insightface.app import FaceAnalysis
		except ImportError as exc:  # pragma: no cover - optional dependency
			raise EmbeddingError("insightface is not installed") from exc

		app = FaceAnalysis(name=self.model_name, providers=self.providers)
		app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
		self._model = app
		return app

	@staticmethod
	def _extract_from_model(model: Any, face_image: np.ndarray) -> np.ndarray:
		if hasattr(model, "get"):
			faces = model.get(face_image)
			if not faces:
				raise EmbeddingError("No face embedding returned by the model")
			face = faces[0]
			embedding = getattr(face, "embedding", None)
			if embedding is None:
				raise EmbeddingError("Model returned a face without an embedding")
			return np.asarray(embedding, dtype=np.float32)

		if callable(model):
			embedding = model(face_image)
			return np.asarray(embedding, dtype=np.float32)

		raise EmbeddingError(f"Unsupported embedding model: {type(model)!r}")

	def extract_embedding(self, face: Any) -> np.ndarray:
		"""Return a normalized embedding vector for the provided face crop."""
		face_image = _as_face_image(face)
		if face_image.ndim == 1:
			return normalize_embedding(face_image)

		model = self._load_model()
		embedding = self._extract_from_model(model, face_image)
		return normalize_embedding(embedding)

	def extract_embedding_from_frame(self, frame: Any) -> tuple[np.ndarray, Any]:
		"""Extract embedding from full frame using ArcFace's internal detection.

		Returns (embedding, detected_face_info) where detected_face_info contains
		the face object from InsightFace with bbox, embedding, etc.
		"""
		frame_image = np.asarray(frame)
		if frame_image.size == 0:
			raise EmbeddingError("Frame must not be empty")

		model = self._load_model()
		faces = model.get(frame_image)
		if not faces:
			raise EmbeddingError("No face detected in frame by ArcFace")

		# Select the most prominent face (largest detection score or biggest bounding box)
		if len(faces) > 1:
			# Sort by detection score (confidence) or bbox area, take the best one
			face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0) or FaceEmbedder._bbox_area(getattr(f, 'bbox', None)))
		else:
			face = faces[0]

		embedding = getattr(face, "embedding", None)
		if embedding is None:
			raise EmbeddingError("ArcFace returned face without embedding")

		return normalize_embedding(embedding), face

	@staticmethod
	def _bbox_area(bbox: Any) -> float:
		"""Calculate bounding box area for face size comparison."""
		if bbox is None or len(bbox) < 4:
			return 0.0
		try:
			# bbox is typically [x1, y1, x2, y2]
			return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
		except Exception:
			return 0.0

	def embed(self, face: Any) -> np.ndarray:
		return self.extract_embedding(face)

	def __call__(self, face: Any) -> np.ndarray:
		return self.extract_embedding(face)

