"""Shared helpers for UI screens."""

from __future__ import annotations

from typing import Any, Callable

import customtkinter as ctk


class ScreenFrame(ctk.CTkFrame):
	def __init__(self, master: Any, title: str, subtitle: str = "", **kwargs: Any) -> None:
		super().__init__(master, **kwargs)
		self.title = title
		self.subtitle = subtitle
		self._build_header()

	def _build_header(self) -> None:
		head = ctk.CTkFrame(self)
		head.pack(fill="x", padx=20, pady=(20, 12))
		ctk.CTkLabel(head, text=self.title, font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w", padx=20, pady=(18, 4))
		if self.subtitle:
			ctk.CTkLabel(head, text=self.subtitle, text_color="#AAB2C0", wraplength=900, justify="left").pack(anchor="w", padx=20, pady=(0, 18))

	def on_show(self) -> None:
		"""Hook called when the screen is displayed."""

	def on_hide(self) -> None:
		"""Hook called when the screen is hidden."""


def camera_frame_source(device_index: int = 0):
	"""Return a callable that yields frames from OpenCV if available."""
	try:
		import cv2
	except Exception as exc:
		def unavailable():
			return None

		unavailable.release = lambda: None  # type: ignore[attr-defined]
		unavailable.error = f"OpenCV import failed: {exc}"  # type: ignore[attr-defined]
		return unavailable

	capture = cv2.VideoCapture(device_index)
	if not capture.isOpened():
		capture.release()
		def unavailable():
			return None

		unavailable.release = lambda: None  # type: ignore[attr-defined]
		unavailable.error = f"Could not open camera at index {device_index}"  # type: ignore[attr-defined]
		return unavailable

	def read_frame():
		success, frame = capture.read()
		return frame if success else None

	def release() -> None:
		capture.release()

	read_frame.release = release  # type: ignore[attr-defined]
	read_frame.error = None  # type: ignore[attr-defined]

	return read_frame


def release_camera_source(frame_source: Callable[[], Any]) -> None:
	"""Release the underlying capture if the source exposes a release hook."""
	try:
		release = getattr(frame_source, "release", None)
		if callable(release):
			release()
	except Exception:
		pass


def camera_source_error(frame_source: Callable[[], Any]) -> str | None:
	"""Return camera open error if the source exposes one."""
	value = getattr(frame_source, "error", None)
	return value if isinstance(value, str) and value else None
