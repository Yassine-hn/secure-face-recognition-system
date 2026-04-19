"""Offline biometric evaluation metrics (FAR/FRR/EER/ROC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class MetricsError(ValueError):
	"""Raised when metric inputs are malformed."""


@dataclass(frozen=True)
class ThresholdMetrics:
	threshold: float
	far: float
	frr: float
	true_accepts: int
	false_accepts: int
	true_rejects: int
	false_rejects: int


@dataclass(frozen=True)
class EERResult:
	eer: float
	threshold: float
	far: float
	frr: float


def _to_scores(name: str, scores: Iterable[float]) -> np.ndarray:
	arr = np.asarray(list(scores), dtype=np.float32)
	if arr.ndim != 1:
		raise MetricsError(f"{name} must be a 1-D sequence")
	if arr.size == 0:
		raise MetricsError(f"{name} must not be empty")
	return arr


def compute_threshold_metrics(
	genuine_scores: Iterable[float],
	impostor_scores: Iterable[float],
	threshold: float,
) -> ThresholdMetrics:
	"""Compute FAR and FRR at a single threshold."""
	genuine = _to_scores("genuine_scores", genuine_scores)
	impostor = _to_scores("impostor_scores", impostor_scores)

	true_accepts = int(np.sum(genuine >= threshold))
	false_rejects = int(genuine.size - true_accepts)
	false_accepts = int(np.sum(impostor >= threshold))
	true_rejects = int(impostor.size - false_accepts)

	frr = false_rejects / float(genuine.size)
	far = false_accepts / float(impostor.size)

	return ThresholdMetrics(
		threshold=float(threshold),
		far=float(far),
		frr=float(frr),
		true_accepts=true_accepts,
		false_accepts=false_accepts,
		true_rejects=true_rejects,
		false_rejects=false_rejects,
	)


def candidate_thresholds(
	genuine_scores: Iterable[float],
	impostor_scores: Iterable[float],
	steps: int = 200,
) -> np.ndarray:
	genuine = _to_scores("genuine_scores", genuine_scores)
	impostor = _to_scores("impostor_scores", impostor_scores)
	minimum = float(min(np.min(genuine), np.min(impostor)))
	maximum = float(max(np.max(genuine), np.max(impostor)))
	if minimum == maximum:
		return np.array([minimum], dtype=np.float32)
	steps = max(2, int(steps))
	return np.linspace(minimum, maximum, steps, dtype=np.float32)


def sweep_threshold_metrics(
	genuine_scores: Iterable[float],
	impostor_scores: Iterable[float],
	steps: int = 200,
) -> list[ThresholdMetrics]:
	thresholds = candidate_thresholds(genuine_scores, impostor_scores, steps=steps)
	return [
		compute_threshold_metrics(genuine_scores, impostor_scores, float(threshold))
		for threshold in thresholds
	]


def compute_eer(
	genuine_scores: Iterable[float],
	impostor_scores: Iterable[float],
	steps: int = 200,
) -> EERResult:
	"""Estimate EER by selecting the threshold where |FAR - FRR| is minimal."""
	metrics = sweep_threshold_metrics(genuine_scores, impostor_scores, steps=steps)
	best = min(metrics, key=lambda item: abs(item.far - item.frr))
	eer = float((best.far + best.frr) / 2.0)
	return EERResult(
		eer=eer,
		threshold=best.threshold,
		far=best.far,
		frr=best.frr,
	)


def roc_curve_points(
	genuine_scores: Iterable[float],
	impostor_scores: Iterable[float],
	steps: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (fpr, tpr, thresholds) suitable for plotting ROC curves."""
	metrics = sweep_threshold_metrics(genuine_scores, impostor_scores, steps=steps)
	fpr = np.asarray([item.far for item in metrics], dtype=np.float32)
	tpr = np.asarray([1.0 - item.frr for item in metrics], dtype=np.float32)
	thresholds = np.asarray([item.threshold for item in metrics], dtype=np.float32)
	order = np.argsort(fpr)
	return fpr[order], tpr[order], thresholds[order]


def roc_auc(fpr: Iterable[float], tpr: Iterable[float]) -> float:
	fpr_arr = _to_scores("fpr", fpr)
	tpr_arr = _to_scores("tpr", tpr)
	if fpr_arr.shape != tpr_arr.shape:
		raise MetricsError("fpr and tpr must have the same shape")
	order = np.argsort(fpr_arr)
	return float(np.trapz(tpr_arr[order], fpr_arr[order]))

