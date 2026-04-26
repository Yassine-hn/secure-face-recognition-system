"""Recognition screen for live face authentication."""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from core.recognition import RecognitionService
from security.audit_log import AuditLogger
from ui.base import ScreenFrame, camera_frame_source, camera_source_error, release_camera_source
from ui.camera_widget import CameraWidget


class RecognitionScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Recognition",
			subtitle="Capture a frame, validate liveness, and match it against enrolled templates.",
			**kwargs,
		)
		self.services = services
		self.camera_source = camera_frame_source()
		self.status_var = ctk.StringVar(value="Ready to scan.")
		self.result_var = ctk.StringVar(value="No recognition performed yet.")

		self._build_body()

	def _build_body(self) -> None:
		body = ctk.CTkFrame(self)
		body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		left = ctk.CTkFrame(body)
		left.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
		self.camera = CameraWidget(left, frame_source=self.camera_source, width=640, height=360)
		self.camera.pack(fill="both", expand=True, padx=20, pady=20)
		self.camera.start()

		right = ctk.CTkFrame(body, width=360)
		right.pack(side="right", fill="y", padx=(10, 20), pady=20)

		ctk.CTkLabel(right, text="Status", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		ctk.CTkLabel(right, textvariable=self.status_var, wraplength=300, justify="left").pack(anchor="w", padx=20)

		ctk.CTkLabel(right, text="Result", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(24, 6))
		ctk.CTkLabel(right, textvariable=self.result_var, wraplength=300, justify="left").pack(anchor="w", padx=20)

		ctk.CTkButton(right, text="Run Recognition", command=self.run_recognition).pack(fill="x", padx=20, pady=(30, 8))
		ctk.CTkButton(right, text="Refresh Frame Source", command=self._refresh_source).pack(fill="x", padx=20, pady=8)

	def _refresh_source(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		self.status_var.set(error or "Frame source refreshed.")

	def on_show(self) -> None:
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		if error:
			self.status_var.set(error)

	def on_hide(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)

	def run_recognition(self) -> None:
		try:
			frame = self.camera_source()
			if frame is None:
				self.status_var.set("No frame available from camera.")
				return

			passive_result = self.services.passive_liveness.classify(frame)
			if not passive_result.is_live:
				self.status_var.set("Passive liveness rejected the frame.")
				self.result_var.set(f"DENIED by passive liveness at score {passive_result.score:.3f}")
				self.services.audit_logger.append_event(
					result="DENIED",
					metadata={"screen": "recognition", "mode": "passive", "score": passive_result.score},
				)
				return

			records = self.services.enrollment.database.list_enrollments()
			templates = {record.student_id: record.template for record in records}
			if not templates:
				self.status_var.set("No enrolled templates found.")
				self.result_var.set("Enroll a student first.")
				return

			result = self.services.recognition.recognize(frame, templates)
			if result.accepted and result.student_id:
				message = f"GRANTED: {result.student_id} at score {result.score:.3f}"
				status = "Recognition succeeded."
			else:
				message = f"DENIED at score {result.score:.3f}"
				status = "Recognition failed."

			self.result_var.set(message)
			self.status_var.set(status)
			self.services.audit_logger.append_event(
				result="GRANTED" if result.accepted else "DENIED",
				student_id=result.student_id,
				metadata={"score": result.score, "passive_score": passive_result.score},
			)
		except Exception as exc:
			self.status_var.set(f"Recognition error: {exc}")
			self.result_var.set("Recognition could not be completed.")
