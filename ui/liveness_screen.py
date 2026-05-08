"""Liveness screen for passive scoring and active challenge orchestration."""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from ui.base import ScreenFrame, camera_frame_source, camera_source_error, release_camera_source
from ui.camera_widget import CameraWidget


class LivenessScreen(ScreenFrame):
	def __init__(self, master: Any, services: Any, **kwargs: Any) -> None:
		super().__init__(
			master,
			title="Liveness",
			subtitle="Inspect passive spoof scoring and the active challenge set used before recognition.",
			**kwargs,
		)
		self.services = services
		self.camera_source = camera_frame_source()
		self.passive_result_var = ctk.StringVar(value="Passive liveness not checked yet.")
		self.challenge_var = ctk.StringVar(value="No active session created yet.")

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

		ctk.CTkLabel(right, text="Passive Liveness", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		ctk.CTkLabel(right, textvariable=self.passive_result_var, wraplength=300, justify="left").pack(anchor="w", padx=20)
		ctk.CTkButton(right, text="Check Current Frame", command=self.check_passive).pack(fill="x", padx=20, pady=(12, 8))

		ctk.CTkLabel(right, text="Active Challenge", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(24, 6))
		ctk.CTkLabel(right, textvariable=self.challenge_var, wraplength=300, justify="left").pack(anchor="w", padx=20)
		ctk.CTkButton(right, text="Generate Session", command=self.generate_session).pack(fill="x", padx=20, pady=(12, 8))

	def check_passive(self) -> None:
		try:
			frame = self.camera_source()
			if frame is None:
				self.passive_result_var.set("No frame available from camera.")
				return

			try:
				detection = self.services.recognition.detect_face(frame)
				face_crop = getattr(detection, "face", frame)
			except Exception:
				face_crop = frame

			result = self.services.passive_liveness.classify(face_crop)
			if not result.used_model:
				display = (
					f"⚠ HEURISTIC FALLBACK — score {result.score:.3f}\n"
					"MiniFASNet model is NOT loaded.\n"
					"This score is UNRELIABLE and cannot distinguish "
					"real faces from photos. Install the "
					"Silent-Face-Anti-Spoofing package and download "
					"the model weights."
				)
			else:
				display = f"{result.label.upper()} at score {result.score:.3f}"
			self.passive_result_var.set(display)
			self.services.audit_logger.append_event(
				result="GRANTED" if result.is_live else "DENIED",
				activity_type="PASSIVE_LIVENESS",
				metadata={
					"screen": "liveness",
					"mode": "passive",
					"score": result.score,
					"used_model": result.used_model,
				},
			)
		except Exception as exc:
			self.passive_result_var.set(f"Passive check failed: {exc}")

	def generate_session(self) -> None:
		try:
			session = self.services.active_liveness.create_session()
			self.challenge_var.set(f"Token: {session.token[:12]}... | Challenges: {', '.join(session.challenge_names)}")
		except Exception as exc:
			self.challenge_var.set(f"Could not create session: {exc}")

	def on_show(self) -> None:
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		if error:
			self.passive_result_var.set(error)

	def on_hide(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
