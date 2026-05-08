"""Iris centroid tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import os

import numpy as np


def _debug_enabled() -> bool:
	return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
	if _debug_enabled():
		print(message)


class IrisDetectionError(ValueError):
	"""Raised when iris inputs are malformed."""


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
	array = np.asarray(list(points), dtype=np.float32)
	if array.ndim != 2 or array.shape[1] != 2:
		raise IrisDetectionError(f"points must be an Nx2 array, got shape {array.shape}")
	if array.shape[0] == 0:
		raise IrisDetectionError("points must not be empty")
	return array


def iris_centroid(points: Iterable[Sequence[float]]) -> np.ndarray:
	"""Return the centroid of the provided iris landmarks."""
	landmarks = _as_points(points)
	return np.mean(landmarks, axis=0).astype(np.float32)


def iris_distance(
	points: Iterable[Sequence[float]],
	target: Sequence[float],
) -> float:
	centroid = iris_centroid(points)
	target_point = np.asarray(target, dtype=np.float32)
	if target_point.shape != (2,):
		raise IrisDetectionError("target must be a 2-D point")
	return float(np.linalg.norm(centroid - target_point))


@dataclass(frozen=True)
class IrisTarget:
	position: tuple[float, float]
	tolerance: float = 15.0


def iris_tracking_detected(
	points: Iterable[Sequence[float]],
	target: Sequence[float],
	tolerance: float = 15.0,
) -> bool:
	if tolerance <= 0.0:
		raise IrisDetectionError("tolerance must be positive")
	return iris_distance(points, target) <= tolerance


def evaluate_iris_tracking(
	points: Iterable[Sequence[float]],
	target: Sequence[float] | IrisTarget | None = None,
	tolerance: float | None = None,
) -> bool:
	"""Evaluate iris tracking against a target point or IrisTarget."""
	if isinstance(target, IrisTarget):
		distance = iris_distance(points, target.position)
		passed = distance <= target.tolerance
		_debug_print(
			f"[DEBUG] evaluate_iris_tracking: distance={distance:.2f}, "
			f"tolerance={target.tolerance:.2f} => {'PASS' if passed else 'FAIL'}"
		)
		return passed
	if target is None:
		raise IrisDetectionError("target must be provided")
	if tolerance is None:
		tolerance = 15.0
	distance = iris_distance(points, target)
	passed = distance <= tolerance
	_debug_print(
		f"[DEBUG] evaluate_iris_tracking: distance={distance:.2f}, "
		f"tolerance={tolerance:.2f} => {'PASS' if passed else 'FAIL'}"
	)
	return passed

