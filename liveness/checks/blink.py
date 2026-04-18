"""Eye Aspect Ratio (EAR) utilities for blink detection."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class BlinkDetectionError(ValueError):
	"""Raised when blink inputs are malformed."""


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
	array = np.asarray(list(points), dtype=np.float32)
	if array.ndim != 2 or array.shape[1] != 2:
		raise BlinkDetectionError(f"points must be an Nx2 array, got shape {array.shape}")
	if array.shape[0] < 6:
		raise BlinkDetectionError("at least 6 points are required")
	return array


def eye_aspect_ratio(points: Iterable[Sequence[float]]) -> float:
	"""Compute EAR using the standard 6-point eye landmark layout."""
	landmarks = _as_points(points)
	a = np.linalg.norm(landmarks[1] - landmarks[5])
	b = np.linalg.norm(landmarks[2] - landmarks[4])
	c = np.linalg.norm(landmarks[0] - landmarks[3])
	if c == 0.0:
		raise BlinkDetectionError("horizontal eye distance must not be zero")
	return float((a + b) / (2.0 * c))


def blink_detected(
	current_ear: float,
	baseline_ear: float,
	drop_ratio: float = 0.30,
) -> bool:
	"""Return True when EAR drops sufficiently below baseline."""
	if baseline_ear <= 0.0:
		raise BlinkDetectionError("baseline_ear must be positive")
	threshold = baseline_ear * (1.0 - drop_ratio)
	return float(current_ear) <= threshold


def evaluate_blink(
	points: Iterable[Sequence[float]],
	baseline: float | None = None,
	threshold: float | None = None,
) -> bool:
	"""Evaluate a blink challenge from eye landmarks and an optional baseline."""
	ear = eye_aspect_ratio(points)
	if threshold is not None:
		return ear <= threshold
	if baseline is None:
		raise BlinkDetectionError("baseline or threshold must be provided")
	return blink_detected(ear, baseline)

