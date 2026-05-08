"""Enrollment screen for capturing and storing a new student template."""

from __future__ import annotations

import traceback
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
		self._face_cascade: Any | None = None
		self._overlay_disabled = False

		self._build_body()

	def _build_body(self) -> None:
		body = ctk.CTkFrame(self)
		body.pack(fill="both", expand=True, padx=20, pady=(0, 20))

		left = ctk.CTkFrame(body)
		left.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
		self.camera = CameraWidget(
			left,
			frame_source=self.camera_source,
			frame_processor=self._overlay_face_bbox,
			width=640,
			height=360,
		)
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

		# Enrolled students list section
		ctk.CTkLabel(right, text="Enrolled Students", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(30, 6))
		self.enrolled_list_frame = ctk.CTkScrollableFrame(right, height=200)
		self.enrolled_list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
		ctk.CTkButton(right, text="Refresh List", command=self._refresh_enrolled_list).pack(fill="x", padx=20, pady=(0, 8))

	def _refresh_source(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		self.status_var.set(error or "Camera source refreshed.")

	def _refresh_enrolled_list(self) -> None:
		"""Refresh the list of enrolled students."""
		# Clear existing widgets
		for widget in self.enrolled_list_frame.winfo_children():
			widget.destroy()

		try:
			records = self.services.enrollment.database.list_enrollments()
			if not records:
				ctk.CTkLabel(self.enrolled_list_frame, text="No students enrolled yet.").pack(pady=10)
				return

			for record in records:
				row = ctk.CTkFrame(self.enrolled_list_frame)
				row.pack(fill="x", pady=2)

				info = ctk.CTkLabel(row, text=record.student_id, anchor="w")
				info.pack(side="left", padx=10, pady=5)

				delete_btn = ctk.CTkButton(
					row,
					text="Delete",
					width=60,
					fg_color="red",
					hover_color="#8B0000",
					command=lambda sid=record.student_id: self._delete_student(sid),
				)
				delete_btn.pack(side="right", padx=10, pady=5)
		except Exception as exc:
			ctk.CTkLabel(self.enrolled_list_frame, text=f"Error loading list: {exc}").pack(pady=10)

	def _delete_student(self, student_id: str) -> None:
		"""Delete an enrolled student after confirmation."""
		try:
			self.services.enrollment.database.delete_enrollment(student_id)
			self.status_var.set(f"Deleted student {student_id}.")
			self.services.audit_logger.append_event(
				result="GRANTED",
				student_id=student_id,
				activity_type="ENROLL",
				metadata={"screen": "enrollment", "action": "delete"},
			)
			self._refresh_enrolled_list()
		except Exception as exc:
			self.status_var.set(f"Failed to delete {student_id}: {exc}")

	def _overlay_face_bbox(self, frame: Any) -> Any:
		if self._overlay_disabled:
			return frame
		try:
			import cv2
		except Exception:
			self._overlay_disabled = True
			return frame

		try:
			if self._face_cascade is None:
				cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
				cascade = cv2.CascadeClassifier(cascade_path)
				if cascade.empty():
					self._overlay_disabled = True
					return frame
				self._face_cascade = cascade

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
		except Exception:
			return frame

		annotated = frame.copy()
		for (x, y, w, h) in faces:
			x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
			cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
			cv2.putText(
				annotated,
				"Face",
				(x1, max(16, y1 - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(60, 220, 60),
				2,
				cv2.LINE_AA,
			)
		return annotated

	def on_show(self) -> None:
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		if error:
			self.status_var.set(error)
		self._refresh_enrolled_list()

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
				activity_type="ENROLL",
				metadata={"screen": "enrollment", "samples": sample_count},
			)
			self.student_id_var.set("")
			self._refresh_enrolled_list()
		except Exception as exc:
			traceback.print_exc()
			self.status_var.set(f"Enrollment failed: {exc}")
