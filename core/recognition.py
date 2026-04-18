"""Recognition pipeline that connects detection, embedding, and matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from core.matcher import FaceMatcher, MatchResult


@dataclass(frozen=True)
class RecognitionResult:
	student_id: Optional[str]
	score: float
	threshold: float
	accepted: bool
	embedding: np.ndarray
	face: Optional[np.ndarray] = None
	match: Optional[MatchResult] = None


def _call_supported(target: Any, method_names: list[str], *args: Any) -> Any:
	if callable(target):
		return target(*args)
	for method_name in method_names:
		method = getattr(target, method_name, None)
		if callable(method):
			return method(*args)
	raise TypeError(f"Unsupported pipeline target: {target!r}")


class RecognitionService:
	"""Run a face through detect -> embed -> match."""

	def __init__(self, detector: Any, embedder: Any, matcher: FaceMatcher) -> None:
		self.detector = detector
		self.embedder = embedder
		self.matcher = matcher

	def detect_face(self, frame: Any) -> Any:
		return _call_supported(self.detector, ["detect_face", "detect", "extract_face"], frame)

	def extract_embedding(self, face: Any) -> np.ndarray:
		embedding = _call_supported(self.embedder, ["extract_embedding", "embed", "encode"], face)
		return np.asarray(embedding, dtype=np.float32)

	def recognize(
		self,
		frame: Any,
		enrolled_templates: Mapping[str, np.ndarray | list[float] | tuple[float, ...]],
	) -> RecognitionResult:
		face = self.detect_face(frame)
		embedding = self.extract_embedding(face)
		match = self.matcher.find_best_match(embedding, enrolled_templates)
		if match is None:
			return RecognitionResult(
				student_id=None,
				score=0.0,
				threshold=self.matcher.threshold,
				accepted=False,
				embedding=embedding,
				face=face,
				match=None,
			)

		student_id = match.student_id if match.is_match else None
		return RecognitionResult(
			student_id=student_id,
			score=match.score,
			threshold=match.threshold,
			accepted=match.is_match,
			embedding=embedding,
			face=face,
			match=match,
		)

	def recognize_frame(
		self,
		frame: Any,
		enrolled_templates: Mapping[str, np.ndarray | list[float] | tuple[float, ...]],
	) -> RecognitionResult:
		return self.recognize(frame, enrolled_templates)
