"""Enrollment screen for capturing and storing a new student template."""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from ui.base import ScreenFrame, camera_frame_source, camera_source_error, release_camera_source
from ui.camera_widget import CameraWidget


class EnrollmentScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Enrollment",
			subtitle="Enroll a student by capturing multiple face samples and storing a transformed template.",
			**kwargs,
		)
		self.services = services
		self.camera_source = camera_frame_source()
		self.status_var = ctk.StringVar(value="Ready to enroll.")
		self.student_id_var = ctk.StringVar(value="")
		self.sample_count_var = ctk.IntVar(value=5)

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

		ctk.CTkLabel(right, text="Student ID", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		ctk.CTkEntry(right, textvariable=self.student_id_var, placeholder_text="e.g. S001").pack(fill="x", padx=20)

		ctk.CTkLabel(right, text="Samples", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		ctk.CTkSlider(right, from_=1, to=10, number_of_steps=9, variable=self.sample_count_var).pack(fill="x", padx=20)
		ctk.CTkLabel(right, textvariable=self.sample_count_var).pack(anchor="w", padx=20, pady=(4, 0))

		ctk.CTkLabel(right, text="Status", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(24, 6))
		ctk.CTkLabel(right, textvariable=self.status_var, wraplength=300, justify="left").pack(anchor="w", padx=20)

		ctk.CTkButton(right, text="Enroll Student", command=self.enroll_student).pack(fill="x", padx=20, pady=(30, 8))
		ctk.CTkButton(right, text="Refresh Camera", command=self._refresh_source).pack(fill="x", padx=20, pady=8)

	def _refresh_source(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		self.status_var.set(error or "Camera source refreshed.")

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

	def enroll_student(self) -> None:
		student_id = self.student_id_var.get().strip()
		if not student_id:
			self.status_var.set("Enter a student ID before enrolling.")
			return

		sample_count = int(self.sample_count_var.get())
		try:
			record = self.services.enrollment.enroll_from_capture(
				student_id=student_id,
				frame_source=self.camera_source,
				face_detector=self.services.detector,
				embedder=self.services.embedder,
				sample_count=sample_count,
				max_attempts=max(20, sample_count * 5),
				metadata={"source": "ui", "samples": sample_count},
			)
			self.status_var.set(f"Enrolled {record.student_id} successfully.")
			self.services.audit_logger.append_event(
				result="GRANTED",
				student_id=student_id,
				metadata={"screen": "enrollment", "samples": sample_count},
			)
		except Exception as exc:
			self.status_var.set(f"Enrollment failed: {exc}")
