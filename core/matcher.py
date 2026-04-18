"""Cosine similarity and threshold-based identity matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


class MatcherError(ValueError):
	"""Raised when matcher inputs are malformed."""


@dataclass(frozen=True)
class MatchResult:
	"""Result of a single identification attempt."""

	student_id: str
	score: float
	threshold: float

	@property
	def is_match(self) -> bool:
		return self.score >= self.threshold


def _as_1d_vector(name: str, vector: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
	arr = np.asarray(vector, dtype=np.float32)
	if arr.ndim != 1:
		raise MatcherError(f"{name} must be a 1-D vector, got shape {arr.shape}")
	if arr.size == 0:
		raise MatcherError(f"{name} must not be empty")
	return arr


def cosine_similarity(
	vector_a: np.ndarray | list[float] | tuple[float, ...],
	vector_b: np.ndarray | list[float] | tuple[float, ...],
) -> float:
	"""Compute cosine similarity in the range [-1, 1]."""
	a = _as_1d_vector("vector_a", vector_a)
	b = _as_1d_vector("vector_b", vector_b)

	if a.shape != b.shape:
		raise MatcherError(f"Vector shape mismatch: {a.shape} vs {b.shape}")

	a_norm = float(np.linalg.norm(a))
	b_norm = float(np.linalg.norm(b))
	if a_norm == 0.0 or b_norm == 0.0:
		raise MatcherError("Cosine similarity is undefined for zero-norm vectors")

	score = float(np.dot(a, b) / (a_norm * b_norm))
	return float(np.clip(score, -1.0, 1.0))


class FaceMatcher:
	"""Identify a face embedding against enrolled template embeddings."""

	def __init__(self, threshold: float = 0.5) -> None:
		self.set_threshold(threshold)

	def set_threshold(self, threshold: float) -> None:
		if not 0.0 <= threshold <= 1.0:
			raise MatcherError("threshold must be within [0.0, 1.0]")
		self.threshold = float(threshold)

	def find_best_match(
		self,
		query_embedding: np.ndarray | list[float] | tuple[float, ...],
		enrolled_templates: Mapping[str, np.ndarray | list[float] | tuple[float, ...]],
	) -> Optional[MatchResult]:
		"""Return the highest-scoring identity candidate, if any templates exist."""
		if not enrolled_templates:
			return None

		query = _as_1d_vector("query_embedding", query_embedding)

		best_student_id: Optional[str] = None
		best_score = -1.0

		for student_id, template in enrolled_templates.items():
			template_vec = _as_1d_vector(f"template[{student_id}]", template)
			score = cosine_similarity(query, template_vec)
			if score > best_score:
				best_student_id = student_id
				best_score = score

		if best_student_id is None:
			return None

		return MatchResult(
			student_id=best_student_id,
			score=best_score,
			threshold=self.threshold,
		)

	def identify(
		self,
		query_embedding: np.ndarray | list[float] | tuple[float, ...],
		enrolled_templates: Mapping[str, np.ndarray | list[float] | tuple[float, ...]],
	) -> tuple[Optional[str], float]:
		"""Return (student_id, score), or (None, score) when below threshold."""
		result = self.find_best_match(query_embedding, enrolled_templates)
		if result is None:
			return None, 0.0
		if result.is_match:
			return result.student_id, result.score
		return None, result.score


def batch_scores(
	query_embedding: np.ndarray | list[float] | tuple[float, ...],
	enrolled_templates: Mapping[str, np.ndarray | list[float] | tuple[float, ...]],
) -> Dict[str, float]:
	"""Compute cosine scores between the query and all enrolled templates."""
	query = _as_1d_vector("query_embedding", query_embedding)
	return {
		student_id: cosine_similarity(query, template)
		for student_id, template in enrolled_templates.items()
	}
