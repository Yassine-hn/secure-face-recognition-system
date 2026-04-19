"""Shared helpers for UI screens."""

from __future__ import annotations

from typing import Any, Optional

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
	except Exception:
		return lambda: None

	capture = cv2.VideoCapture(device_index)
	if not capture.isOpened():
		capture.release()
		return lambda: None

	def read_frame():
		success, frame = capture.read()
		return frame if success else None

	return read_frame
