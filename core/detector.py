"""RetinaFace-based face detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


class DetectionError(RuntimeError):
	"""Raised when face detection fails or invalid inputs are supplied."""


@dataclass(frozen=True)
class DetectionResult:
	bbox: tuple[int, int, int, int]
	score: float
	face: np.ndarray
	embedding: Optional[np.ndarray] = None
	landmarks: Optional[np.ndarray] = None


def _as_image(frame: Any) -> np.ndarray:
	image = np.asarray(frame)
	if image.ndim not in (2, 3):
		raise DetectionError(f"frame must be a 2-D or 3-D image, got shape {image.shape}")
	if image.size == 0:
		raise DetectionError("frame must not be empty")
	return image


def crop_face(image: np.ndarray, bbox: Sequence[float], margin: float = 0.0) -> np.ndarray:
	"""Crop a face region from an image using an absolute bbox or normalized bbox."""
	frame = _as_image(image)
	if len(bbox) != 4:
		raise DetectionError("bbox must contain exactly 4 values")

	height, width = frame.shape[:2]
	x1, y1, x2, y2 = [float(value) for value in bbox]
	if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
		x1 *= width
		x2 *= width
		y1 *= height
		y2 *= height

	box_width = max(1.0, x2 - x1)
	box_height = max(1.0, y2 - y1)
	pad_x = box_width * margin
	pad_y = box_height * margin
	left = max(0, int(np.floor(x1 - pad_x)))
	top = max(0, int(np.floor(y1 - pad_y)))
	right = min(width, int(np.ceil(x2 + pad_x)))
	bottom = min(height, int(np.ceil(y2 + pad_y)))

	if left >= right or top >= bottom:
		raise DetectionError("computed crop is empty")
	return frame[top:bottom, left:right].copy()


class FaceDetector:
	"""Detect the most prominent face in a frame.

	The class uses InsightFace/RetinaFace when available and falls back to injected
	callables or precomputed detections for tests and lightweight environments.
	"""

	def __init__(
		self,
		model: Any | None = None,
		model_name: str = "buffalo_l",
		providers: Optional[list[str]] = None,
		ctx_id: int = -1,
		det_size: tuple[int, int] = (640, 640),
		margin: float = 0.15,
	) -> None:
		self._model = model
		self.model_name = model_name
		self.providers = providers
		self.ctx_id = ctx_id
		self.det_size = det_size
		self.margin = margin

	def _load_model(self) -> Any:
		if self._model is not None:
			return self._model

		try:
			from insightface.app import FaceAnalysis

			app = FaceAnalysis(name=self.model_name, providers=self.providers)
			app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
			self._model = app
			return app
		except Exception:
			# Fallback to OpenCV Haar Cascade when InsightFace is not installed or available
			try:
				import cv2

				cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
				classifier = cv2.CascadeClassifier(cascade_path)

				class _OpenCVHaarModel:
					def __init__(self, clf: Any) -> None:
						self.clf = clf

					def get(self, frame: np.ndarray) -> list[Any]:
						if frame.ndim == 3:
							gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						else:
							gray = frame
						rects = self.clf.detectMultiScale(
							gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
						)
						faces = []
						for x, y, w, h in rects:
							class _FaceBox:
								def __init__(self, bbox_arr: np.ndarray) -> None:
									self.bbox = bbox_arr
									self.det_score = 0.95
									self.kps = None

							faces.append(
								_FaceBox(
									np.array([x, y, x + w, y + h], dtype=np.float32)
								)
							)
						return faces

				self._model = _OpenCVHaarModel(classifier)
				return self._model
			except Exception as exc:
				raise DetectionError(
					"Neither insightface nor opencv cascade could be loaded for face detection"
				) from exc

	def _get_faces(self, frame: np.ndarray) -> list[Any]:
		model = self._load_model()
		if hasattr(model, "get"):
			faces = model.get(frame)
			return list(faces or [])
		if callable(model):
			faces = model(frame)
			return list(faces or [])
		raise DetectionError(f"Unsupported detection model: {type(model)!r}")

	@staticmethod
	def _select_best_face(faces: Sequence[Any]) -> Any:
		if not faces:
			raise DetectionError("No face detected")

		def score(face: Any) -> float:
			candidate_score = getattr(face, "det_score", None)
			if candidate_score is None:
				candidate_score = getattr(face, "score", 0.0)
			return float(candidate_score)

		return max(faces, key=score)

	def detect(self, frame: Any) -> DetectionResult:
		image = _as_image(frame)
		faces = self._get_faces(image)
		face = self._select_best_face(faces)
		bbox = getattr(face, "bbox", None)
		if bbox is None:
			raise DetectionError("Detected face does not provide a bounding box")
		cropped = crop_face(image, bbox, margin=self.margin)
		score = float(getattr(face, "det_score", getattr(face, "score", 0.0)))
		embedding = getattr(face, "embedding", None)
		if embedding is not None:
			embedding = np.asarray(embedding, dtype=np.float32)
		landmarks = getattr(face, "landmark", None)
		if landmarks is not None:
			landmarks = np.asarray(landmarks)
		return DetectionResult(
			bbox=tuple(int(round(value)) for value in np.asarray(bbox).tolist()),
			score=score,
			face=cropped,
			embedding=embedding,
			landmarks=landmarks,
		)

	def detect_face(self, frame: Any) -> np.ndarray:
		"""Return a cropped face image for embedding extraction."""
		return self.detect(frame).face

	def __call__(self, frame: Any) -> np.ndarray:
		return self.detect_face(frame)

