"""Recognition screen for live face authentication."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import traceback
from typing import Any

import customtkinter as ctk

from core.recognition import RecognitionService
from security.audit_log import AuditLogger
from ui.base import ScreenFrame, camera_frame_source, camera_source_error, release_camera_source
from ui.camera_widget import CameraWidget

# Thresholds for triggering active liveness verification
PASSIVE_LIVENESS_UNCERTAIN_LOW = 0.4
PASSIVE_LIVENESS_UNCERTAIN_HIGH = 0.7
RECOGNITION_AMBIGUOUS_LOW = 0.5
RECOGNITION_AMBIGUOUS_HIGH = 0.7


def _debug_enabled() -> bool:
	return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
	if _debug_enabled():
		print(message)


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
		self.timer_var = ctk.StringVar(value="Time remaining: --")
		self._face_cascade: Any | None = None
		self._overlay_disabled = False
		self._active_liveness_required = False
		self._active_liveness_button: ctk.CTkButton | None = None
		self._active_challenge_start: float | None = None
		self._active_challenge_window: float | None = None
		self._current_challenge: str | None = None
		self._iris_direction: str | None = None
		self._iris_baseline: tuple[float, float] | None = None
		self._iris_eye_width: float | None = None
		self._mirror_preview = True
		self._iris_direction_threshold = 0.10
		self._iris_mirror_mode = self._mirror_preview
		self._smile_baseline: float | None = None
		self._timer_job: str | None = None

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

		ctk.CTkLabel(right, text="Status", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 6))
		ctk.CTkLabel(right, textvariable=self.status_var, wraplength=300, justify="left").pack(anchor="w", padx=20)
		ctk.CTkLabel(right, textvariable=self.timer_var).pack(anchor="w", padx=20, pady=(6, 0))

		ctk.CTkLabel(right, text="Result", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(24, 6))
		ctk.CTkLabel(right, textvariable=self.result_var, wraplength=300, justify="left").pack(anchor="w", padx=20)

		ctk.CTkButton(right, text="Run Recognition", command=self.run_recognition).pack(fill="x", padx=20, pady=(30, 8))
		self._active_liveness_button = ctk.CTkButton(
			right,
			text="Start Active Liveness Check",
			command=self.run_active_liveness,
			fg_color="orange",
			hover_color="#FF8C00",
		)
		# Initially hidden, shown when active liveness is required
		self._active_liveness_button.pack_forget()
		ctk.CTkButton(right, text="Refresh Frame Source", command=self._refresh_source).pack(fill="x", padx=20, pady=8)

	def _refresh_source(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
		self.camera_source = camera_frame_source()
		self.camera.frame_source = self.camera_source
		self.camera.start()
		error = camera_source_error(self.camera_source)
		self.status_var.set(error or "Frame source refreshed.")

	def _overlay_face_bbox(self, frame: Any) -> Any:
		if self._overlay_disabled:
			return frame
		try:
			import cv2
		except Exception:
			self._overlay_disabled = True
			return frame

		display_frame = frame
		if self._mirror_preview:
			display_frame = cv2.flip(frame, 1)

		try:
			if self._face_cascade is None:
				cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
				cascade = cv2.CascadeClassifier(cascade_path)
				if cascade.empty():
					self._overlay_disabled = True
					return frame
				self._face_cascade = cascade

			gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
			faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
		except Exception:
			return frame

		annotated = display_frame.copy()
		if self._current_challenge == "iris" and self._iris_direction is not None:
			h, w = annotated.shape[:2]
			center_x = int(w * 0.5)
			center_y = int(h * 0.18)
			arrow_len = max(60, int(w * 0.2))
			direction = self._iris_direction
			if direction == "left":
				start = (center_x + arrow_len // 2, center_y)
				end = (center_x - arrow_len // 2, center_y)
				label = "LOOK LEFT"
			else:
				start = (center_x - arrow_len // 2, center_y)
				end = (center_x + arrow_len // 2, center_y)
				label = "LOOK RIGHT"
			cv2.arrowedLine(annotated, start, end, (30, 30, 220), 4, tipLength=0.25)
			cv2.putText(
				annotated,
				label,
				(center_x - arrow_len // 2, max(20, center_y - 12)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(240, 240, 240),
				2,
				cv2.LINE_AA,
			)
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

	def on_hide(self) -> None:
		self.camera.stop()
		release_camera_source(self.camera_source)
		self._stop_timer()

	def _check_active_liveness_required(
		self,
		passive_score: float,
		recognition_score: float,
		is_recognized: bool,
	) -> tuple[bool, str]:
		"""Determine if active liveness verification is required.

		Returns (required: bool, reason: str).
		"""
		# Uncertain passive liveness: passed but with low confidence
		if PASSIVE_LIVENESS_UNCERTAIN_LOW <= passive_score < PASSIVE_LIVENESS_UNCERTAIN_HIGH:
			return True, "uncertain_passive_liveness"

		# Ambiguous recognition: face detected but weak match
		if is_recognized and RECOGNITION_AMBIGUOUS_LOW <= recognition_score < RECOGNITION_AMBIGUOUS_HIGH:
			return True, "ambiguous_recognition_match"

		return False, ""

	def _show_active_liveness_button(self, reason: str) -> None:
		"""Show the active liveness check button when additional verification is needed."""
		self._active_liveness_required = True
		self._active_liveness_button.pack(fill="x", padx=20, pady=8, before=self._active_liveness_button.master.winfo_children()[-1])
		self.status_var.set(f"Additional verification required: {reason.replace('_', ' ')}")

	def _hide_active_liveness_button(self) -> None:
		"""Hide the active liveness check button."""
		self._active_liveness_required = False
		self._active_liveness_button.pack_forget()

	def _get_challenge_instruction(self, challenge_name: str) -> str:
		"""Get user-friendly instruction for a challenge."""
		instructions = {
			"blink": "BLINK YOUR EYES\n\nPlease blink your eyes naturally.\nThe system will detect the eye movement.",
			"iris": "LOOK LEFT OR RIGHT\n\nMove only your eyes in the arrow direction.\nKeep your head still.",
			"head_pose": "TURN YOUR HEAD\n\nSlowly turn or tilt your head.\nMove left, right, up, or down.",
			"mouth": "OPEN YOUR MOUTH\n\nOpen your mouth wide.\nThe system will detect the mouth opening.",
			"smile": "SMILE\n\nGive a natural smile.\nShow your teeth if possible.",
		}
		if challenge_name == "iris":
			direction = (self._iris_direction or "left").upper()
			return (
				f"LOOK {direction}\n\n"
				"Move only your eyes in the arrow direction.\n"
				"Keep your head still."
			)
		return instructions.get(challenge_name, f"Complete challenge: {challenge_name}")

	def run_active_liveness(self) -> None:
		"""Run active liveness challenge-response verification with sequential UI."""
		try:
			self._hide_active_liveness_button()
			self.status_var.set("Starting active liveness verification...")
			self.result_var.set("Initializing challenges...")

			# Create session with exactly 2 challenges
			session = self.services.active_liveness.create_session()
			challenges = list(session.challenge_names)

			if len(challenges) < 2:
				# Ensure at least 2 challenges for meaningful verification
				all_challenges = ["blink", "head_pose", "mouth", "smile", "iris"]
				while len(challenges) < 2:
					remaining = [c for c in all_challenges if c not in challenges]
					if remaining:
						challenges.append(remaining[0])

			per_challenge_window = float(session.window_seconds)
			self._active_challenge_window = per_challenge_window
			self._active_challenge_start = None
			self._stop_timer()
			_debug_print(
				f"[DEBUG] Active liveness session started: token={session.token[:12]}..., "
				f"per_challenge={per_challenge_window}, challenges={challenges}"
			)

			# Log start
			self.services.audit_logger.append_event(
				result="PENDING",
				activity_type="ACTIVE_LIVENESS",
				session_token=session.token,
				metadata={"challenges": challenges, "status": "started"},
			)

			# Show first challenge
			self._run_challenge_sequence(challenges, session.token, 0, [])

		except Exception as exc:
			traceback.print_exc()
			self.status_var.set(f"Active liveness error: {exc}")
			self.result_var.set("Active liveness check could not be completed.")

	def _run_challenge_sequence(
		self,
		challenges: list[str],
		session_token: str,
		current_index: int,
		results: list[bool],
	) -> None:
		"""Run challenges sequentially with real MediaPipe-based verification."""
		if current_index >= len(challenges):
			# All challenges completed
			passed_count = sum(results)
			total = len(results)
			all_passed = passed_count == total

			self.services.audit_logger.append_event(
				result="GRANTED" if all_passed else "DENIED",
				activity_type="ACTIVE_LIVENESS",
				session_token=session_token,
				metadata={"results": results, "passed": passed_count, "total": total},
			)

			if all_passed:
				student_id = getattr(self, "_expected_student_id", "Unknown")
				self.result_var.set(
					f"ACTIVE LIVENESS PASSED ({passed_count}/{total})\n\n"
					f"Identity verified during challenges.\n"
					f"ACCESS GRANTED - Student ID: {student_id}"
				)
				self.status_var.set("Active liveness verified - Access granted.")
				self.services.audit_logger.append_event(
					result="GRANTED",
					student_id=student_id,
					activity_type="RECOGNIZE",
					metadata={"reason": "active_liveness_passed"},
				)
			else:
				self.result_var.set(
					f"ACTIVE LIVENESS FAILED ({passed_count}/{total})\n\n"
					"Some challenges failed or identity mismatch.\n"
					"Please try again or contact support."
				)
				self.status_var.set("Active liveness failed - Access denied.")
			self._active_challenge_start = None
			self._active_challenge_window = None
			self._current_challenge = None
			self._iris_direction = None
			self._iris_baseline = None
			self._iris_eye_width = None
			self._stop_timer()
			return

		challenge = challenges[current_index]
		self._current_challenge = challenge
		if challenge == "iris":
			self._iris_direction = random.choice(["left", "right"])
			self._iris_baseline = None
			self._iris_eye_width = None
			_debug_print(f"[DEBUG] Iris direction set: {self._iris_direction}")
		else:
			self._iris_direction = None
			self._iris_baseline = None
			self._iris_eye_width = None
		self._active_challenge_start = time.monotonic()
		self._stop_timer()
		self._update_timer()
		instruction = self._get_challenge_instruction(challenge)
		_debug_print(
			f"[DEBUG] Active liveness: starting challenge {challenge} "
			f"({current_index + 1}/{len(challenges)}), per_challenge={self._active_challenge_window}"
		)

		# Display challenge instruction
		progress = f"Challenge {current_index + 1} of {len(challenges)}"
		self.result_var.set(f"{progress}\n\n{instruction}\n\nPlease perform the action now...")
		self.status_var.set(f"Active liveness: {challenge.replace('_', ' ').title()} - Detecting...")

		# Pre-fetch enrollment data in main thread (SQLite is not thread-safe)
		records = self.services.enrollment.database.list_enrollments()
		templates = {record.student_id: record.template for record in records}

		# Run real frame capture + evaluation in a background thread
		import threading
		thread = threading.Thread(
			target=self._evaluate_challenge_from_frames,
			args=(challenge, challenges, session_token, current_index, results, templates),
			daemon=True,
		)
		thread.start()

	def _active_time_remaining(self) -> float | None:
		if self._active_challenge_start is None or self._active_challenge_window is None:
			return None
		elapsed = time.monotonic() - self._active_challenge_start
		return max(0.0, self._active_challenge_window - elapsed)

	def _stop_timer(self) -> None:
		if self._timer_job is not None:
			try:
				self.after_cancel(self._timer_job)
			except Exception:
				pass
			self._timer_job = None
		self.timer_var.set("Time remaining: --")

	def _update_timer(self) -> None:
		remaining = self._active_time_remaining()
		if remaining is None:
			self.timer_var.set("Time remaining: --")
			self._timer_job = None
			return
		self.timer_var.set(f"Time remaining: {remaining:.1f}s")
		if remaining <= 0.0:
			self._timer_job = None
			return
		self._timer_job = self.after(200, self._update_timer)

	# ---- MediaPipe landmark indices used by the check modules ----
	# Eyelid landmarks for EAR (6 points per eye, using left eye)
	_LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
	_RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
	# Head pose: left eye center, right eye center, nose tip, left mouth, right mouth, chin
	_HEAD_POSE_INDICES = [33, 263, 1, 61, 291, 199]
	# Inner lip landmarks for MAR (8 points)
	_INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 308, 415, 310, 311, 312, 13, 82, 81, 80]
	# Simplified inner lip 8-point layout: corners + top/bottom
	_MOUTH_MAR_INDICES = [61, 84, 17, 314, 405, 320, 307, 375]  # Standard 8-point mouth for MAR
	# Iris landmarks (MediaPipe 468-477)
	_LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
	_RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
	# Eye corner indices for width estimation
	_LEFT_EYE_CORNER_INDICES = (33, 133)
	_RIGHT_EYE_CORNER_INDICES = (362, 263)

	def _extract_landmarks_2d(self, face_landmarks: Any, indices: list[int], w: int, h: int) -> list[list[float]]:
		"""Extract (x, y) pixel coordinates for the given landmark indices."""
		points = []
		for idx in indices:
			lm = face_landmarks.landmark[idx]
			points.append([lm.x * w, lm.y * h])
		return points

	def _estimate_eye_width(self, face_landmarks: Any, w: int, h: int) -> float:
		left_outer = face_landmarks.landmark[self._LEFT_EYE_CORNER_INDICES[0]]
		left_inner = face_landmarks.landmark[self._LEFT_EYE_CORNER_INDICES[1]]
		right_outer = face_landmarks.landmark[self._RIGHT_EYE_CORNER_INDICES[0]]
		right_inner = face_landmarks.landmark[self._RIGHT_EYE_CORNER_INDICES[1]]
		left_dx = (left_outer.x - left_inner.x) * w
		left_dy = (left_outer.y - left_inner.y) * h
		right_dx = (right_outer.x - right_inner.x) * w
		right_dy = (right_outer.y - right_inner.y) * h
		left_width = (left_dx ** 2 + left_dy ** 2) ** 0.5
		right_width = (right_dx ** 2 + right_dy ** 2) ** 0.5
		width = (left_width + right_width) / 2.0
		return float(width)

	def _evaluate_challenge_from_frames(
		self,
		challenge_name: str,
		challenges: list[str],
		session_token: str,
		current_index: int,
		results: list[bool],
		templates: dict[str, np.ndarray],
	) -> None:
		"""Capture frames within the session time budget and run liveness evaluation."""

		try:
			import mediapipe as mp
		except ImportError:
			_debug_print(f"[DEBUG] evaluate_{challenge_name}: MediaPipe not available")
			# MediaPipe not available — challenge cannot be evaluated, fail
			self.after(0, lambda: self._verify_challenge(
				challenges, session_token, current_index, results, passed=False,
			))
			return

		mp_face_mesh = mp.solutions.face_mesh
		# Use refine_landmarks=True to get iris landmarks (indices 468-477)
		face_mesh = mp_face_mesh.FaceMesh(
			static_image_mode=False,
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)

		challenge_passed = False
		baseline_captured = False
		baseline_ear: float | None = None
		baseline_pose: Any = None
		per_challenge = self._active_challenge_window or 3.0
		capture_duration = per_challenge
		start_time = time.monotonic()
		consecutive_valid_frames = 0
		MIN_CONSECUTIVE_FOR_BASELINE = 5
		MIN_FRAMES_BEFORE_PASS = 10
		received_frame = False
		warned_no_frames = False

		if capture_duration is None or capture_duration <= 0.0:
			_debug_print(f"[DEBUG] evaluate_{challenge_name}: no time remaining for capture")
			self.after(0, lambda: self._verify_challenge(
				challenges, session_token, current_index, results, passed=False,
			))
			return

		_debug_print(
			f"[DEBUG] evaluate_{challenge_name}: capture_duration={capture_duration:.2f}s "
			f"(per_challenge={per_challenge:.2f}s)"
		)

		try:
			import cv2
		except ImportError:
			_debug_print(f"[DEBUG] evaluate_{challenge_name}: OpenCV not available")
			self.after(0, lambda: self._verify_challenge(
				challenges, session_token, current_index, results, passed=False,
			))
			return

		frames_processed = 0
		mouth_baseline_samples: list[float] = []
		head_pose_samples: list[Any] = []
		last_iris_debug_time = 0.0
		last_head_pose_debug_time = 0.0

		while (time.monotonic() - start_time) < capture_duration:
			frame = None
			try:
				frame = self.camera_source()
			except Exception:
				pass

			if frame is None:
				if not warned_no_frames and (time.monotonic() - start_time) > 1.0:
					_debug_print(f"[DEBUG] evaluate_{challenge_name}: no frames received from camera")
					warned_no_frames = True
				time.sleep(0.05)
				continue
			if not received_frame:
				func_name = {
					"blink": "evaluate_blink",
					"head_pose": "evaluate_head_pose",
					"mouth": "evaluate_mouth_open",
					"smile": "evaluate_smile",
					"iris": "evaluate_iris_tracking",
				}.get(challenge_name, f"evaluate_{challenge_name}")
				_debug_print(f"[DEBUG] {func_name}: received stream of frames successfully")
				received_frame = True

			# Convert BGR -> RGB for MediaPipe
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			mp_result = face_mesh.process(rgb_frame)

			if not mp_result.multi_face_landmarks:
				# No face detected in this frame
				consecutive_valid_frames = 0
				time.sleep(0.05)
				continue

			face_landmarks = mp_result.multi_face_landmarks[0]
			h, w = frame.shape[:2]
			frames_processed += 1
			consecutive_valid_frames += 1

			try:
				if challenge_name == "blink":
					from liveness.checks.blink import eye_aspect_ratio, blink_detected
					left_pts = self._extract_landmarks_2d(face_landmarks, self._LEFT_EYE_EAR_INDICES, w, h)
					right_pts = self._extract_landmarks_2d(face_landmarks, self._RIGHT_EYE_EAR_INDICES, w, h)
					left_ear = eye_aspect_ratio(left_pts)
					right_ear = eye_aspect_ratio(right_pts)
					avg_ear = (left_ear + right_ear) / 2.0

					if not baseline_captured and consecutive_valid_frames >= MIN_CONSECUTIVE_FOR_BASELINE:
						# Sanity check: EAR should be in physiological range for open eyes
						if 0.15 <= avg_ear <= 0.35:
							baseline_ear = avg_ear
							baseline_captured = True
							_debug_print(f"[DEBUG] evaluate_blink: baseline EAR={avg_ear:.3f}")
					elif baseline_ear is not None and baseline_ear > 0:
						if frames_processed >= MIN_FRAMES_BEFORE_PASS and blink_detected(avg_ear, baseline_ear, drop_ratio=0.30):
							_debug_print(
								f"[DEBUG] evaluate_blink: EAR={avg_ear:.3f}, baseline={baseline_ear:.3f}, "
								"drop_ratio=0.30 => PASS"
							)
							challenge_passed = True

				elif challenge_name == "head_pose":
					from liveness.checks.head_pose import HeadPose, pose_from_landmarks, relative_pose_delta, head_pose_within_threshold
					pts = self._extract_landmarks_2d(face_landmarks, self._HEAD_POSE_INDICES, w, h)
					current_pose = pose_from_landmarks(pts)

					if not baseline_captured and consecutive_valid_frames >= MIN_CONSECUTIVE_FOR_BASELINE:
						# Sanity check: pose values should be reasonable
						if all(abs(v) <= 60.0 for v in (current_pose.pitch, current_pose.yaw, current_pose.roll)):
							head_pose_samples.append(current_pose)
							if len(head_pose_samples) >= 10:
								baseline_pose = HeadPose(
									pitch=float(np.median([p.pitch for p in head_pose_samples])),
									yaw=float(np.median([p.yaw for p in head_pose_samples])),
									roll=float(np.median([p.roll for p in head_pose_samples])),
								)
								baseline_captured = True
								_debug_print(
									f"[DEBUG] evaluate_head_pose: baseline pitch={baseline_pose.pitch:.1f}, "
									f"yaw={baseline_pose.yaw:.1f}, roll={baseline_pose.roll:.1f}"
								)
					elif baseline_pose is not None:
						if frames_processed >= MIN_FRAMES_BEFORE_PASS:
							delta = relative_pose_delta(current_pose, baseline_pose)
							threshold = 7.0
							if head_pose_within_threshold(delta, threshold=threshold):
								_debug_print(
									f"[DEBUG] evaluate_head_pose: delta pitch={delta.pitch:.1f}, "
									f"yaw={delta.yaw:.1f}, roll={delta.roll:.1f}, threshold={threshold:.1f} => PASS"
								)
								challenge_passed = True
							elif _debug_enabled():
								current_time = time.monotonic()
								if current_time - last_head_pose_debug_time >= 0.5:
									_debug_print(
										f"[DEBUG] evaluate_head_pose: delta pitch={delta.pitch:.1f}, "
										f"yaw={delta.yaw:.1f}, roll={delta.roll:.1f}, threshold={threshold:.1f} => not yet"
									)
									last_head_pose_debug_time = current_time

				elif challenge_name == "mouth":
					from liveness.checks.mouth import mouth_aspect_ratio, mouth_open
					pts = self._extract_landmarks_2d(face_landmarks, self._MOUTH_MAR_INDICES, w, h)
					mar = mouth_aspect_ratio(pts)
					_debug_print(
						f"[DEBUG] evaluate_mouth_open: frame={frames_processed}, MAR={mar:.3f}, "
						f"frames_needed={MIN_FRAMES_BEFORE_PASS}, consecutive={consecutive_valid_frames}"
					)
					
					# Capture baseline as the median of initial samples
					if not baseline_captured and consecutive_valid_frames >= MIN_CONSECUTIVE_FOR_BASELINE:
						mouth_baseline_samples.append(mar)
						if len(mouth_baseline_samples) >= 10:
							baseline_value = float(np.median(np.asarray(mouth_baseline_samples)))
							self._mouth_baseline = baseline_value
							baseline_captured = True
							_debug_print(
								f"[DEBUG] evaluate_mouth_open: baseline MAR (median)={baseline_value:.3f}"
							)
					elif baseline_captured and hasattr(self, '_mouth_baseline'):
						# Check if mouth is open enough compared to baseline
						mar_increase = mar - self._mouth_baseline
						_debug_print(
							f"[DEBUG] evaluate_mouth_open: MAR increase={mar_increase:.3f} "
							f"(current={mar:.3f}, baseline={self._mouth_baseline:.3f})"
						)
						if frames_processed >= MIN_FRAMES_BEFORE_PASS and mar_increase >= 0.10:
							_debug_print(
								f"[DEBUG] evaluate_mouth_open: PASS (MAR increase={mar_increase:.3f} >= 0.10)"
							)
							challenge_passed = True
					elif not baseline_captured and frames_processed >= MIN_FRAMES_BEFORE_PASS:
						# Fallback: accept a clearly open mouth even without baseline
						if mar >= 0.55:
							_debug_print(
								f"[DEBUG] evaluate_mouth_open: PASS (absolute MAR={mar:.3f} >= 0.55)"
							)
							challenge_passed = True

				elif challenge_name == "smile":
					from liveness.checks.smile import lip_corner_spread_ratio, smile_detected
					# smile needs all 468+ landmarks for index-based access
					all_pts = self._extract_landmarks_2d(
						face_landmarks,
						list(range(min(len(face_landmarks.landmark), 468))),
						w, h,
					)
					if len(all_pts) > max(291, 454):
						ratio = lip_corner_spread_ratio(all_pts)
						if self._smile_baseline is None and consecutive_valid_frames >= MIN_CONSECUTIVE_FOR_BASELINE:
							self._smile_baseline = ratio
							_debug_print(
								f"[DEBUG] evaluate_smile: baseline ratio={ratio:.3f}"
							)
						elif self._smile_baseline is not None and frames_processed >= MIN_FRAMES_BEFORE_PASS:
							delta = ratio - self._smile_baseline
							threshold = 0.05
							if delta >= threshold:
								_debug_print(
									f"[DEBUG] evaluate_smile: ratio={ratio:.3f}, baseline={self._smile_baseline:.3f}, "
									f"delta={delta:.3f} >= {threshold:.2f} => PASS"
								)
								challenge_passed = True
							elif _debug_enabled() and frames_processed % 10 == 0:
								_debug_print(
									f"[DEBUG] evaluate_smile: ratio={ratio:.3f}, baseline={self._smile_baseline:.3f}, "
									f"delta={delta:.3f} < {threshold:.2f}"
								)

				elif challenge_name == "iris":
					from liveness.checks.iris import iris_centroid
					left_iris_pts = self._extract_landmarks_2d(face_landmarks, self._LEFT_IRIS_INDICES, w, h)
					right_iris_pts = self._extract_landmarks_2d(face_landmarks, self._RIGHT_IRIS_INDICES, w, h)

					left_c = iris_centroid(left_iris_pts)
					right_c = iris_centroid(right_iris_pts)
					current_center = ((left_c[0] + right_c[0]) / 2.0, (left_c[1] + right_c[1]) / 2.0)

					if not baseline_captured and consecutive_valid_frames >= MIN_CONSECUTIVE_FOR_BASELINE:
						eye_width = self._estimate_eye_width(face_landmarks, w, h)
						if eye_width > 0.0:
							self._iris_baseline = current_center
							self._iris_eye_width = eye_width
							baseline_captured = True
							_debug_print(
								f"[DEBUG] evaluate_iris_tracking: baseline={self._iris_baseline}, "
								f"eye_width={eye_width:.2f}"
							)
					elif baseline_captured and self._iris_baseline and self._iris_eye_width:
						dx = current_center[0] - self._iris_baseline[0]
						threshold = self._iris_eye_width * self._iris_direction_threshold
						direction = self._iris_direction or "left"
						if self._iris_mirror_mode:
							direction = "right" if direction == "left" else "left"
						if frames_processed >= MIN_FRAMES_BEFORE_PASS:
							passed = dx <= -threshold if direction == "left" else dx >= threshold
							if passed:
								_debug_print(
									f"[DEBUG] evaluate_iris_tracking: dx={dx:.2f}, "
									f"threshold={threshold:.2f}, direction={direction} => PASS"
								)
								challenge_passed = True
							elif _debug_enabled():
								current_time = time.monotonic()
								if current_time - last_iris_debug_time >= 0.5:
									_debug_print(
										f"[DEBUG] evaluate_iris_tracking: dx={dx:.2f}, "
										f"threshold={threshold:.2f}, direction={direction} => not yet"
									)
									last_iris_debug_time = current_time

			except Exception as exc:
				_debug_print(f"[DEBUG] evaluate_{challenge_name}: evaluation error: {exc}")
				# If a single frame fails evaluation, continue to next frame
				pass

			if challenge_passed:
				# Bind identity to the frame where liveness passed
				try:
					rec_result = self.services.recognition.recognize(frame, templates)
					expected_id = getattr(self, "_expected_student_id", None)
					
					if not rec_result.accepted or rec_result.student_id != expected_id:
						challenge_passed = False
						_debug_print(
							f"[DEBUG] Active liveness identity mismatch. Expected {expected_id}, got {rec_result.student_id}"
						)
				except Exception as e:
					_debug_print(f"[DEBUG] Active liveness recognition failed: {e}")
					challenge_passed = False

				break

			time.sleep(0.05)

		face_mesh.close()

		# Schedule UI update on the main thread
		self.after(0, lambda: self._verify_challenge(
			challenges, session_token, current_index, results, passed=challenge_passed,
		))

	def _verify_challenge(
		self,
		challenges: list[str],
		session_token: str,
		current_index: int,
		results: list[bool],
		passed: bool,
	) -> None:
		"""Verify current challenge and advance to next."""
		challenge = challenges[current_index]
		results.append(passed)

		# Show success/failure message
		status = "✓ PASSED" if passed else "✗ FAILED"
		self.result_var.set(
			f"Challenge {current_index + 1} {status}\n\n"
			f"{challenge.replace('_', ' ').title()} challenge completed.\n\n"
			"Proceeding to next challenge..."
		)

		# Log individual challenge result
		self.services.audit_logger.append_event(
			result="GRANTED" if passed else "DENIED",
			activity_type="ACTIVE_LIVENESS",
			session_token=session_token,
			metadata={"challenge": challenge, "index": current_index, "passed": passed},
		)

		# Advance to next challenge after brief delay
		self.after(1500, lambda: self._run_challenge_sequence(
			challenges, session_token, current_index + 1, results
		))

	def run_recognition(self) -> None:
		try:
			# Hide any previously shown active liveness button
			self._hide_active_liveness_button()
			self.status_var.set("Capturing frames...")
			self.result_var.set("Analyzing liveness and identity...")

			# Pre-fetch enrollment data in main thread (SQLite is not thread-safe)
			records = self.services.enrollment.database.list_enrollments()
			templates = {record.student_id: record.template for record in records}

			# Run in a background thread to avoid freezing UI
			import threading
			threading.Thread(
				target=self._run_recognition_thread,
				args=(templates,),
				daemon=True
			).start()
		except Exception as exc:
			traceback.print_exc()
			self.status_var.set(f"Recognition error: {exc}")
			self.result_var.set("Recognition could not be completed.")

	def _run_recognition_thread(self, templates: dict[str, np.ndarray]) -> None:
		import time
		import numpy as np
		from liveness.passive import PassiveLivenessResult

		try:
			frames = []
			for _ in range(3):
				f = self.camera_source()
				if f is not None:
					frames.append(f)
				time.sleep(0.1)

			if not frames:
				self.after(0, lambda: self.status_var.set("No frames available from camera."))
				return

			if not templates:
				self.after(0, lambda: self.status_var.set("No enrolled templates found."))
				self.after(0, lambda: self.result_var.set("Enroll a student first."))
				return

			passive_scores = []
			best_result = None
			used_model = False

			for frame in frames:
				# Get a cropped face for passive liveness
				try:
					detection = self.services.recognition.detect_face(frame)
					face_crop = getattr(detection, "face", frame)
				except Exception:
					face_crop = frame

				# Run passive liveness
				p_res = self.services.passive_liveness.classify(face_crop)
				passive_scores.append(p_res.score)
				used_model = p_res.used_model

				# Run recognition
				rec_res = self.services.recognition.recognize(frame, templates)
				if best_result is None or rec_res.score > best_result.score:
					best_result = rec_res

			avg_passive_score = float(np.mean(passive_scores))
			
			passive_result = PassiveLivenessResult(
				label="live" if avg_passive_score >= self.services.passive_liveness.threshold else "spoof",
				score=avg_passive_score,
				threshold=self.services.passive_liveness.threshold,
				used_model=used_model
			)

			self.after(0, lambda r=best_result, p=passive_result: self._handle_recognition_result(r, p))
		except Exception as exc:
			traceback.print_exc()
			self.after(0, lambda e=exc: self.status_var.set(f"Recognition error: {e}"))
			self.after(0, lambda: self.result_var.set("Recognition could not be completed."))

	def _handle_recognition_result(self, result: Any, passive_result: Any) -> None:
		try:
			# Check if embedding extraction failed
			if result.error:
				self.status_var.set(f"Recognition failed: {result.error}")
				self.result_var.set("ACCESS DENIED - Face detection succeeded but embedding extraction failed. Please try again with better lighting or face positioning.")
				self.services.audit_logger.append_event(
					result="DENIED",
					activity_type="RECOGNIZE",
					metadata={"screen": "recognition", "error": result.error, "reason": "embedding_extraction_failed"},
				)
				return

			# Determine outcome based on both passive liveness and recognition results
			liveness_passed = passive_result.is_live
			recognition_accepted = result.accepted and result.student_id is not None

			# Case 1: Recognition denies → DENIED (regardless of liveness)
			if not recognition_accepted:
				if result.student_id is None:
					message = "ACCESS DENIED - Face not recognized (unknown person)"
					denial_reason = "face_not_recognized"
				else:
					message = f"ACCESS DENIED - Known face ({result.student_id}) but below confidence threshold ({result.score:.2%})"
					denial_reason = "below_threshold"
				self.result_var.set(message)
				self.status_var.set("Recognition failed.")
				self.services.audit_logger.append_event(
					result="DENIED",
					student_id=result.student_id,
					activity_type="RECOGNIZE",
					metadata={"score": result.score, "passive_score": passive_result.score, "reason": denial_reason},
				)
				return

			# Case 2: Recognition passes → Active liveness always required
			reason = "active_liveness_required"
			self._expected_student_id = result.student_id
			self._show_active_liveness_button(reason)
			self.result_var.set(
				f"ACTIVE LIVENESS REQUIRED FOR ALL ATTEMPTS.\n"
				f"Avg Passive liveness: {passive_result.score:.3f}\n"
				f"Best Recognition: {result.score:.2%} for student {result.student_id}\n"
				"Click 'Start Active Liveness Check' to proceed."
			)
			self.status_var.set("Active liveness required for this attempt.")
			self.services.audit_logger.append_event(
				result="PENDING",
				student_id=result.student_id,
				activity_type="RECOGNIZE",
				metadata={
					"score": result.score,
					"passive_score": passive_result.score,
					"requires_active_liveness": True,
					"reason": reason,
					"liveness_passed": liveness_passed,
				},
			)
			return
		except Exception as exc:
			traceback.print_exc()
			self.status_var.set(f"Recognition error: {exc}")
			self.result_var.set("Recognition could not be completed.")
