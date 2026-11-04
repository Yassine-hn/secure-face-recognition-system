"""Passive liveness detection based on texture analysis."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
	return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
	if _debug_enabled():
		print(message)


class PassiveLivenessError(RuntimeError):
	"""Raised when passive liveness estimation fails."""


@dataclass(frozen=True)
class PassiveLivenessResult:
	label: str
	score: float
	threshold: float
	used_model: bool = True

	@property
	def is_live(self) -> bool:
		return self.label.lower() == "live"


def _as_image(frame: Any) -> np.ndarray:
	image = np.asarray(frame)
	if image.ndim not in (2, 3):
		raise PassiveLivenessError(f"frame must be a 2-D or 3-D image, got shape {image.shape}")
	if image.size == 0:
		raise PassiveLivenessError("frame must not be empty")
	return image.astype(np.float32)


def _heuristic_live_score(face_image: np.ndarray) -> float:
	"""Cheap fallback that prefers textured, non-uniform crops.

	WARNING: This heuristic is NOT a real anti-spoofing model. It only
	examines basic texture variance and gradient energy, which produces
	nearly identical scores (~0.63) for both real faces and photos.
	It exists solely to keep the module importable in environments without
	MiniFASNet weights — it must NOT be relied upon for security.
	"""
	logger.warning(
		"Using heuristic fallback for passive liveness — scores are UNRELIABLE. "
		"Install Silent-Face-Anti-Spoofing and download MiniFASNet weights."
	)
	gray = face_image.mean(axis=2) if face_image.ndim == 3 else face_image
	variance = float(np.var(gray))
	gy, gx = np.gradient(gray.astype(np.float32))
	energy = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0
	base_score = 0.6 * np.tanh(variance / 500.0) + 0.4 * np.tanh(energy / 20.0)

	# Detect screen moire / scanline frequency disparity
	ratio = float((np.mean(np.abs(gy)) + 1e-4) / (np.mean(np.abs(gx)) + 1e-4))
	if ratio > 2.0:
		base_score *= (1.5 / ratio)
	elif ratio < 0.35:
		base_score *= (ratio / 0.35)

	return float(np.clip(base_score, 0.0, 1.0))


class PassiveLivenessChecker:
	"""Classify a face crop as live or spoof.

	If a model is supplied, it is used first. Otherwise a deterministic heuristic
	is applied so the module remains testable in environments without weights.
	"""

	def __init__(
		self,
		model: Any | None = None,
		threshold: float = 0.5,
		live_label: str = "live",
		spoof_label: str = "spoof",
	) -> None:
		self.model = model
		self.threshold = float(threshold)
		self.live_label = live_label
		self.spoof_label = spoof_label

	def _load_model(self) -> Any:
		if self.model is not None:
			return self.model

		try:
			from liveness.silent_face_wrapper import MiniFASNet
			model = MiniFASNet()
			_debug_print("[DEBUG] Passive liveness: MiniFASNet loaded successfully.")
			return model
		except (ImportError, RuntimeError) as e:
			logger.warning(
				"Could not load MiniFASNet model: %s. "
				"Passive liveness will use an UNRELIABLE heuristic fallback. "
				"Please clone Silent-Face-Anti-Spoofing to models/silent_face/ "
				"and download the MiniFASNet model weights.",
				e
			)
			_debug_print(f"[DEBUG] Passive liveness: MiniFASNet load failed: {e}")
			return None

	@property
	def model_loaded(self) -> bool:
		"""Return True if a real anti-spoofing model is available."""
		return self._load_model() is not None

	@staticmethod
	def _score_from_model(model: Any, face_image: np.ndarray) -> float:
		if hasattr(model, "predict"):
			prediction = model.predict(face_image)
		else:
			prediction = model(face_image)

		array = np.asarray(prediction, dtype=np.float32)
		if array.size == 0:
			raise PassiveLivenessError("model prediction is empty")
		flat = array.reshape(-1)
		if flat.size == 1:
			return float(flat[0])

		# MiniFASNet typically returns [spoof_prob, live_prob]. Use index 1 by default.
		live_index = 1
		if live_index >= flat.size:
			live_index = flat.size - 1
		_debug_print(
			f"[DEBUG] Passive liveness model raw output: {flat.tolist()} "
			f"(using index {live_index} as live probability)"
		)
		return float(flat[live_index])

	def score(self, face_crop: Any) -> tuple[float, bool]:
		"""Return (score, used_model) for the given face crop."""
		image = _as_image(face_crop)
		model = self._load_model()
		if model is None:
			# Use heuristic score when model is not available
			score = _heuristic_live_score(image)
			_debug_print(f"[DEBUG] Passive liveness: using heuristic score={score:.3f}")
			return score, False
		try:
			raw_score = self._score_from_model(model, image)
		except Exception as exc:
			_debug_print(f"[DEBUG] Passive liveness: model call failed: {exc}")
			raise
		score = float(np.clip(raw_score, 0.0, 1.0))
		_debug_print(f"[DEBUG] Passive liveness: model score={score:.3f}")
		return score, True

	def classify(self, face_crop: Any) -> PassiveLivenessResult:
		raw_score, used_model = self.score(face_crop)
		label = self.live_label if raw_score >= self.threshold else self.spoof_label
		if not used_model:
			logger.warning(
				"Passive liveness score %.3f was produced by the HEURISTIC "
				"fallback, not MiniFASNet. This score is unreliable.",
				raw_score,
			)
			_debug_print(
				f"[DEBUG] Passive liveness: used_model={used_model}, score={raw_score:.3f}, threshold={self.threshold:.3f}"
			)
		return PassiveLivenessResult(
			label=label, score=raw_score, threshold=self.threshold, used_model=used_model,
		)

	def is_live(self, face_crop: Any) -> bool:
		result = self.classify(face_crop)
		if not result.used_model:
			logger.warning(
				"is_live() returning %s based on HEURISTIC fallback — "
				"result is NOT trustworthy.",
				result.is_live,
			)
		return result.is_live

	def __call__(self, face_crop: Any) -> PassiveLivenessResult:
		return self.classify(face_crop)

	# Alias for convenience
	evaluate = classify

