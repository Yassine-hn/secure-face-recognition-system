"""Threaded bridge from a frame source into a CustomTkinter image label."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import customtkinter as ctk
from PIL import Image


class CameraWidget(ctk.CTkFrame):
	"""Display frames from a callable source without blocking the UI thread."""

	def __init__(
		self,
		master: Any,
		frame_source: Callable[[], Any],
		frame_processor: Optional[Callable[[Any], Any]] = None,
		fps: int = 12,
		width: int = 640,
		height: int = 360,
		placeholder_text: str = "Camera feed unavailable",
		**kwargs: Any,
	) -> None:
		super().__init__(master, **kwargs)
		self.frame_source = frame_source
		self.frame_processor = frame_processor
		self.fps = max(1, int(fps))
		self.width = width
		self.height = height
		self.placeholder_text = placeholder_text
		self._running = False
		self._worker: threading.Thread | None = None
		self._image_ref: Any = None

		self.preview = ctk.CTkLabel(self, text=self.placeholder_text, width=width, height=height)
		self.preview.pack(fill="both", expand=True)

	def start(self) -> None:
		if self._running:
			return
		self._running = True
		self._worker = threading.Thread(target=self._loop, daemon=True)
		self._worker.start()

	def stop(self) -> None:
		self._running = False

	def destroy(self) -> None:
		self.stop()
		super().destroy()

	def _loop(self) -> None:
		delay = 1.0 / self.fps
		while self._running:
			frame = None
			try:
				frame = self.frame_source()
			except Exception:
				frame = None

			if frame is not None:
				if self.frame_processor is not None:
					try:
						frame = self.frame_processor(frame)
					except Exception:
						pass
				self.after(0, self._update_frame, frame)
			time.sleep(delay)

	def _update_frame(self, frame: Any) -> None:
		if frame is None:
			return

		try:
			from numpy import ndarray
		except Exception:
			ndarray = tuple()  # type: ignore[assignment]

		image = frame
		if hasattr(frame, "shape"):
			try:
				from cv2 import cvtColor, COLOR_BGR2RGB
				image = cvtColor(frame, COLOR_BGR2RGB)
			except Exception:
				image = frame

		try:
			pil_image = Image.fromarray(image)
		except Exception:
			return

		photo = ctk.CTkImage(pil_image, size=(self.width, self.height))
		self._image_ref = photo
		self.preview.configure(image=photo, text="")

