"""Passive liveness detection based on texture analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


class PassiveLivenessError(RuntimeError):
	"""Raised when passive liveness estimation fails."""


@dataclass(frozen=True)
class PassiveLivenessResult:
	label: str
	score: float
	threshold: float

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
	"""Cheap fallback that prefers textured, non-uniform crops."""
	gray = face_image.mean(axis=2) if face_image.ndim == 3 else face_image
	variance = float(np.var(gray))
	energy = float(np.mean(np.abs(np.gradient(gray.astype(np.float32)))))
	raw_score = 0.6 * np.tanh(variance / 500.0) + 0.4 * np.tanh(energy / 20.0)
	return float(np.clip(raw_score, 0.0, 1.0))


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
			from silent_face import MiniFASNet  # type: ignore
		except ImportError:
			return None

		return MiniFASNet()

	@staticmethod
	def _score_from_model(model: Any, face_image: np.ndarray) -> float:
		if hasattr(model, "predict"):
			prediction = model.predict(face_image)
		else:
			prediction = model(face_image)

		array = np.asarray(prediction, dtype=np.float32)
		if array.ndim == 0:
			return float(array)
		if array.size == 1:
			return float(array.reshape(()))
		if array.size >= 2:
			return float(array.flat[0])
		raise PassiveLivenessError("model prediction is empty")

	def score(self, face_crop: Any) -> float:
		image = _as_image(face_crop)
		model = self._load_model()
		if model is None:
			return _heuristic_live_score(image)
		return float(np.clip(self._score_from_model(model, image), 0.0, 1.0))

	def classify(self, face_crop: Any) -> PassiveLivenessResult:
		score = self.score(face_crop)
		label = self.live_label if score >= self.threshold else self.spoof_label
		return PassiveLivenessResult(label=label, score=score, threshold=self.threshold)

	def is_live(self, face_crop: Any) -> bool:
		return self.classify(face_crop).is_live

	def __call__(self, face_crop: Any) -> PassiveLivenessResult:
		return self.classify(face_crop)

