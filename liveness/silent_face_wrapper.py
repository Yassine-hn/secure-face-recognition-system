"""Wrapper for Silent-Face-Anti-Spoofing MiniFASNet model."""

from __future__ import annotations

from collections import OrderedDict
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from liveness.minifasnet_model import (
	CropImage,
	MiniFASNetV2,
	get_kernel,
	parse_model_name,
	to_tensor,
)

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
	return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
	if _debug_enabled():
		print(message)


class SilentFaceWrapper:
	"""Wrapper for Silent-Face-Anti-Spoofing MiniFASNet model."""

	def __init__(self, model_dir: Optional[str] = None, device_id: int = 0) -> None:
		"""Initialize the wrapper.

		Args:
			model_dir: Directory containing the Silent-Face-Anti-Spoofing model weights.
			device_id: GPU device ID (0 for first GPU, -1 for CPU).
		"""
		if model_dir:
			p = Path(model_dir)
			if not p.is_absolute():
				p = Path(__file__).resolve().parent.parent / p
			self.model_dir = p
		else:
			self.model_dir = Path(__file__).resolve().parent.parent / "models" / "silent_face"

		self.device = torch.device(
			f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
		)
		self.model: Optional[torch.nn.Module] = None
		self._model_loaded = False
		self.input_size = (80, 80)
		self.scale = 2.7
		self.crop_image = CropImage()

		self._try_load_model()

	def _try_load_model(self) -> None:
		"""Load the MiniFASNet model weights."""
		try:
			_debug_print(
				f"[DEBUG] SilentFaceWrapper: loading model. model_dir={self.model_dir}, device={self.device}"
			)

			model_file = self.model_dir / "2.7_80x80_MiniFASNetV2.pth"
			if not model_file.exists():
				logger.warning("MiniFASNet model not found at %s", model_file)
				_debug_print(f"[DEBUG] SilentFaceWrapper: model file missing at {model_file}")
				return

			model_name = model_file.name
			h_input, w_input, _model_type, scale = parse_model_name(model_name)
			kernel_size = get_kernel(h_input, w_input)

			# Initialize native model
			self.model = MiniFASNetV2(conv6_kernel=kernel_size).to(self.device)

			# Load weights and strip potential 'module.' prefix
			state_dict = torch.load(str(model_file), map_location=self.device)
			new_state_dict = OrderedDict()
			for key, value in state_dict.items():
				name_key = key[7:] if key.startswith("module.") else key
				new_state_dict[name_key] = value

			self.model.load_state_dict(new_state_dict)
			self.model.eval()
			self._model_loaded = True
			self.input_size = (w_input, h_input)
			self.scale = scale if scale is not None else 2.7

			logger.info("MiniFASNet model loaded successfully from %s", model_file)
			_debug_print(
				f"[DEBUG] SilentFaceWrapper: model loaded, input_size={w_input}x{h_input}, scale={self.scale}"
			)
		except Exception as e:
			logger.error("Failed to load MiniFASNet model: %s", e)
			_debug_print(f"[DEBUG] SilentFaceWrapper: load failed: {e}")

	@property
	def is_available(self) -> bool:
		"""Check if the model is loaded and ready for inference."""
		return self._model_loaded and self.model is not None

	def predict(
		self,
		face_or_frame: np.ndarray,
		bbox: Optional[Sequence[float | int]] = None,
	) -> np.ndarray:
		"""Predict liveness score for a face crop or full frame.

		Args:
			face_or_frame: Face crop or full video frame as numpy array (BGR or RGB).
			bbox: Optional bounding box [x1, y1, x2, y2] in the full frame. When supplied,
			      an expanded patch (scale 2.7) is cropped matching training geometry.

		Returns:
			numpy array of class probabilities [class0_spoof, class1_real, class2_spoof].
		"""
		if not self.is_available or self.model is None:
			raise RuntimeError("MiniFASNet model is not available")

		image = np.asarray(face_or_frame)
		if image.dtype != np.uint8:
			if image.max() <= 1.0:
				image = (image * 255.0).astype(np.uint8)
			else:
				image = image.astype(np.uint8)

		w_input, h_input = self.input_size

		# If bounding box is supplied and input is a full frame, apply scale 2.7 crop
		if (
			bbox is not None
			and len(bbox) == 4
			and image.ndim == 3
			and image.shape[0] > h_input
			and image.shape[1] > w_input
		):
			x1, y1, x2, y2 = [float(v) for v in bbox]
			# Convert (x1, y1, x2, y2) to [x, y, w, h] for CropImage
			box_w = max(1.0, x2 - x1)
			box_h = max(1.0, y2 - y1)
			bbox_xywh = [x1, y1, box_w, box_h]
			input_patch = self.crop_image.crop(
				image, bbox_xywh, scale=self.scale, out_w=w_input, out_h=h_input, crop=True
			)
		else:
			# Fallback: direct resize of face crop
			input_patch = cv2.resize(image, (w_input, h_input))

		# Convert BGR to tensor 1xCxHxW
		tensor = to_tensor(input_patch).unsqueeze(0).to(self.device)

		with torch.no_grad():
			logits = self.model(tensor)
			probabilities = F.softmax(logits, dim=-1).cpu().numpy()

		return probabilities[0]


_silent_face_instance: Optional[SilentFaceWrapper] = None


def get_silent_face_model(
	model_dir: Optional[str] = None, device_id: int = 0
) -> Optional[SilentFaceWrapper]:
	"""Get or create a SilentFaceWrapper singleton instance."""
	global _silent_face_instance
	if _silent_face_instance is None:
		_silent_face_instance = SilentFaceWrapper(model_dir, device_id)
	return _silent_face_instance if _silent_face_instance.is_available else None


class MiniFASNet:
	"""Compatibility class mimicking the expected MiniFASNet interface."""

	def __init__(self, model_dir: Optional[str] = None, device_id: int = 0) -> None:
		self.wrapper = get_silent_face_model(model_dir, device_id)
		if self.wrapper is None:
			raise RuntimeError("MiniFASNet model is not available")

	def predict(
		self,
		face_or_frame: np.ndarray,
		bbox: Optional[Sequence[float | int]] = None,
	) -> np.ndarray:
		"""Predict liveness probabilities."""
		if self.wrapper is None:
			raise RuntimeError("MiniFASNet model is not available")
		return self.wrapper.predict(face_or_frame, bbox=bbox)
