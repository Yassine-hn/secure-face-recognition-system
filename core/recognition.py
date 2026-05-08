"""Recognition pipeline that connects detection, embedding, and matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from core.embedder import EmbeddingError
from core.matcher import FaceMatcher, MatchResult, batch_scores, cosine_similarity
from security.cancelable import CancelableTransformer


@dataclass(frozen=True)
class RecognitionResult:
	student_id: Optional[str]
	score: float
	threshold: float
	accepted: bool
	embedding: Optional[np.ndarray] = None
	face: Optional[np.ndarray] = None
	match: Optional[MatchResult] = None
	error: Optional[str] = None


def _call_supported(target: Any, method_names: list[str], *args: Any) -> Any:
	for method_name in method_names:
		method = getattr(target, method_name, None)
		if callable(method):
			return method(*args)
	if callable(target):
		return target(*args)
	raise TypeError(f"Unsupported pipeline target: {target!r}")


class RecognitionService:
	"""Run a face through detect -> embed -> match."""

	def __init__(
		self,
		detector: Any,
		embedder: Any,
		matcher: FaceMatcher,
		transformer: CancelableTransformer | None = None,
	) -> None:
		self.detector = detector
		self.embedder = embedder
		self.matcher = matcher
		self.transformer = transformer

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
		# Try to extract embedding using ArcFace directly on the full frame first
		# This is more reliable than detecting then cropping separately
		embedding: np.ndarray | None = None
		face: Any | None = None
		extraction_error: str | None = None
		extraction_method = "unknown"

		try:
			# Check if embedder supports full-frame extraction
			if hasattr(self.embedder, "extract_embedding_from_frame"):
				embedding, face = self.embedder.extract_embedding_from_frame(frame)
				extraction_method = "ArcFace full-frame"
			else:
				# Fallback to detect-then-embed pipeline
				detection = self.detect_face(frame)
				direct_embedding = getattr(detection, "embedding", None)
				face = getattr(detection, "face", detection)

				if direct_embedding is not None:
					embedding = np.asarray(direct_embedding, dtype=np.float32)
					extraction_method = "detector direct embedding"
				else:
					embedding = self.extract_embedding(face)
					extraction_method = "detect-then-embed"
		except EmbeddingError as exc:
			extraction_error = str(exc)
			# Try fallback method if available
			try:
				detection = self.detect_face(frame)
				face = getattr(detection, "face", detection)
				embedding = self.extract_embedding(face)
				extraction_error = None  # Fallback succeeded
				extraction_method = "fallback detect-then-embed"
			except EmbeddingError:
				pass  # Keep original error

		print(f"\n[DEBUG] Extraction method: {extraction_method}")
		if extraction_error:
			print(f"[DEBUG] Extraction error: {extraction_error}")

		# Apply cancelable transformation to query embedding (same as enrollment)
		if self.transformer is not None and embedding is not None:
			embedding = self.transformer.transform(embedding)
			print(f"[DEBUG] Applied cancelable transformation")

		if extraction_error or embedding is None:
			return RecognitionResult(
				student_id=None,
				score=0.0,
				threshold=self.matcher.threshold,
				accepted=False,
				embedding=None,
				face=face,
				match=None,
				error=f"Embedding extraction failed: {extraction_error}" if extraction_error else "Unknown error",
			)

		match = self.matcher.find_best_match(embedding, enrolled_templates)

		# DEBUG: Print all scores and match details
		print("\n" + "="*60)
		print("RECOGNITION DEBUG")
		print("="*60)
		print(f"Enrolled students: {list(enrolled_templates.keys())}")
		print(f"Matcher threshold: {self.matcher.threshold}")

		if embedding is not None:
			all_scores = batch_scores(embedding, enrolled_templates)
			print("\nSimilarity scores (cosine similarity, range [-1, 1]):")
			for sid, score in sorted(all_scores.items(), key=lambda x: -x[1]):
				marker = " <-- BEST MATCH" if match and sid == match.student_id else ""
				status = "PASS" if score >= self.matcher.threshold else "FAIL"
				print(f"  {sid}: {score:.4f} [{status}]{marker}")

		if match is None:
			print("\nResult: NO MATCH FOUND (no enrolled templates)")
			print("="*60)
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
		print(f"\nBest match: {match.student_id} (score: {match.score:.4f})")
		print(f"Is match: {match.is_match} (score >= threshold: {match.score:.4f} >= {match.threshold})")
		print(f"Final result: {'RECOGNIZED' if match.is_match else 'REJECTED (below threshold)'} -> student_id={student_id}")
		print("="*60)

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
