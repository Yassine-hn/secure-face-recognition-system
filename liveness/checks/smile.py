"""Smile detection utilities based on lip corner spread."""

from __future__ import annotations

from typing import Iterable, Sequence

import os

import numpy as np


def _debug_enabled() -> bool:
	return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
	if _debug_enabled():
		print(message)


class SmileDetectionError(ValueError):
	"""Raised when smile inputs are malformed."""


def lip_corner_spread_ratio(
	landmarks: Iterable[Sequence[float]],
	left_index: int = 61,
	right_index: int = 291,
	face_width_indices: tuple[int, int] = (234, 454),
) -> float:
	points = np.asarray(list(landmarks), dtype=np.float32)
	if points.ndim != 2 or points.shape[1] != 2:
		raise SmileDetectionError(f"landmarks must be an Nx2 array, got shape {points.shape}")
	if points.shape[0] <= max(left_index, right_index, *face_width_indices):
		raise SmileDetectionError("not enough landmarks for smile computation")

	left = points[left_index]
	right = points[right_index]
	face_left = points[face_width_indices[0]]
	face_right = points[face_width_indices[1]]
	face_width = np.linalg.norm(face_left - face_right)
	if face_width == 0.0:
		raise SmileDetectionError("face width must not be zero")
	return float(np.linalg.norm(left - right) / face_width)


def smile_detected(current_ratio: float, threshold: float) -> bool:
	if threshold <= 0.0:
		raise SmileDetectionError("threshold must be positive")
	return float(current_ratio) >= threshold


def evaluate_smile(
	landmarks: Iterable[Sequence[float]],
	threshold: float | None = None,
) -> bool:
	"""Evaluate the smile challenge from face landmarks."""
	ratio = lip_corner_spread_ratio(landmarks)
	if threshold is None:
		threshold = 0.35
	passed = smile_detected(ratio, threshold)
	_debug_print(f"[DEBUG] evaluate_smile: ratio={ratio:.3f}, threshold={threshold:.3f} => {'PASS' if passed else 'FAIL'}")
	return passed

