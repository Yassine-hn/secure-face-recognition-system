"""Cancelable biometric template utilities based on random projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class CancelableTransformError(ValueError):
	"""Raised when cancelable transform inputs are malformed."""


def _as_vector(name: str, embedding: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
	arr = np.asarray(embedding, dtype=np.float32)
	if arr.ndim != 1:
		raise CancelableTransformError(f"{name} must be a 1-D vector, got shape {arr.shape}")
	if arr.size == 0:
		raise CancelableTransformError(f"{name} must not be empty")
	return arr


def _as_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
	arr = np.asarray(matrix, dtype=np.float32)
	if arr.ndim != 2:
		raise CancelableTransformError(f"{name} must be a 2-D matrix, got shape {arr.shape}")
	if arr.shape[0] != arr.shape[1]:
		raise CancelableTransformError(f"{name} must be square, got shape {arr.shape}")
	return arr


def generate_projection_matrix(embedding_dim: int = 512, seed: int | None = None) -> np.ndarray:
	"""Generate an orthonormal projection matrix of shape (d, d)."""
	if embedding_dim <= 0:
		raise CancelableTransformError("embedding_dim must be positive")

	rng = np.random.default_rng(seed)
	gaussian = rng.normal(loc=0.0, scale=1.0, size=(embedding_dim, embedding_dim)).astype(np.float32)

	# QR gives an orthonormal basis that preserves vector scale and improves numeric stability.
	q, _ = np.linalg.qr(gaussian)
	return q.astype(np.float32)


def apply_cancelable_transform(
	embedding: np.ndarray | list[float] | tuple[float, ...],
	projection_matrix: np.ndarray,
) -> np.ndarray:
	"""Transform a raw embedding into a cancelable template."""
	emb = _as_vector("embedding", embedding)
	proj = _as_matrix("projection_matrix", projection_matrix)

	if emb.shape[0] != proj.shape[0]:
		raise CancelableTransformError(
			f"Embedding dimension {emb.shape[0]} does not match matrix dimension {proj.shape[0]}"
		)

	transformed = proj @ emb
	return transformed.astype(np.float32)


def apply_cancelable_transform_batch(
	embeddings: Iterable[np.ndarray | list[float] | tuple[float, ...]],
	projection_matrix: np.ndarray,
) -> list[np.ndarray]:
	"""Apply cancelable transform to a batch of embeddings."""
	return [apply_cancelable_transform(embedding, projection_matrix) for embedding in embeddings]


@dataclass
class CancelableTransformer:
	"""Encapsulates projection matrix lifecycle for template transformation."""

	embedding_dim: int = 512
	seed: int | None = None

	def __post_init__(self) -> None:
		self._projection_matrix = generate_projection_matrix(self.embedding_dim, self.seed)

	@property
	def projection_matrix(self) -> np.ndarray:
		return self._projection_matrix.copy()

	def transform(self, embedding: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
		return apply_cancelable_transform(embedding, self._projection_matrix)

	def transform_batch(
		self,
		embeddings: Iterable[np.ndarray | list[float] | tuple[float, ...]],
	) -> list[np.ndarray]:
		return apply_cancelable_transform_batch(embeddings, self._projection_matrix)

	def rotate(self, seed: int | None = None) -> np.ndarray:
		"""Rotate templates by issuing a new projection matrix."""
		self.seed = seed
		self._projection_matrix = generate_projection_matrix(self.embedding_dim, seed)
		return self.projection_matrix
