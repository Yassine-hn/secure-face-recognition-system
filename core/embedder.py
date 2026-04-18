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

	def embed(self, face: Any) -> np.ndarray:
		return self.extract_embedding(face)

	def __call__(self, face: Any) -> np.ndarray:
		return self.extract_embedding(face)

