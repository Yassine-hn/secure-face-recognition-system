"""Mouth Aspect Ratio (MAR) utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class MouthDetectionError(ValueError):
	"""Raised when mouth inputs are malformed."""


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
	array = np.asarray(list(points), dtype=np.float32)
	if array.ndim != 2 or array.shape[1] != 2:
		raise MouthDetectionError(f"points must be an Nx2 array, got shape {array.shape}")
	if array.shape[0] < 8:
		raise MouthDetectionError("at least 8 points are required")
	return array


def mouth_aspect_ratio(points: Iterable[Sequence[float]]) -> float:
	"""Compute MAR using the standard 8-point mouth landmark layout."""
	landmarks = _as_points(points)
	a = np.linalg.norm(landmarks[2] - landmarks[6])
	b = np.linalg.norm(landmarks[3] - landmarks[5])
	c = np.linalg.norm(landmarks[0] - landmarks[4])
	if c == 0.0:
		raise MouthDetectionError("horizontal mouth distance must not be zero")
	return float((a + b) / (2.0 * c))


def mouth_open(
	current_mar: float,
	threshold: float,
) -> bool:
	if threshold <= 0.0:
		raise MouthDetectionError("threshold must be positive")
	return float(current_mar) >= threshold


def evaluate_mouth_open(
	points: Iterable[Sequence[float]],
	threshold: float | None = None,
) -> bool:
	"""Evaluate the mouth-open challenge from landmarks."""
	mar = mouth_aspect_ratio(points)
	if threshold is None:
		threshold = 0.5
	return mouth_open(mar, threshold)

