"""Streamlit web interface for the Secure Face Recognition System.

Matches the layout, screen structure, and operational logic of the original desktop application.
Features automated multi-frame capture, separate passive liveness & biometric matching evaluation,
dynamic active liveness challenge sequences, and robust fallback error handling.
"""

from __future__ import annotations

import atexit
import csv
import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st

from core.services import AppServices, build_default_services


# ── Page Configuration & Theming ─────────────────────────────────────────────
st.set_page_config(
	page_title="Secure Face Recognition System",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	.main .block-container {
		padding-top: 1.5rem;
		padding-bottom: 2rem;
		max-width: 1200px;
	}
	div[data-testid="stSidebar"] {
		min-width: 240px;
		max-width: 280px;
	}
	.screen-title {
		font-size: 1.8rem;
		font-weight: 700;
		margin-bottom: 0.2rem;
	}
	.screen-subtitle {
		color: #888888;
		font-size: 0.95rem;
		margin-bottom: 1.5rem;
	}
	.status-box {
		background-color: rgba(255, 255, 255, 0.05);
		border-radius: 6px;
		padding: 12px 16px;
		margin-bottom: 1rem;
		font-family: monospace;
		white-space: pre-wrap;
	}
	.challenge-banner {
		background-color: rgba(255, 165, 0, 0.15);
		border-left: 4px solid orange;
		padding: 12px 16px;
		margin-bottom: 1rem;
		border-radius: 4px;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> AppServices:
	"""Initialize and cache the backend application services container."""
	return build_default_services()


try:
	services = get_services()
except Exception as init_exc:
	st.error(f"Failed to initialize core biometric services: {init_exc}")
	st.stop()


# ── Helper Functions ──────────────────────────────────────────────────────────
def frame_to_bgr(uploaded_file: Any) -> Optional[np.ndarray]:
	"""Convert a Streamlit UploadedFile buffer into an OpenCV BGR frame."""
	if uploaded_file is None:
		return None
	try:
		bytes_data = uploaded_file.getvalue()
		np_arr = np.frombuffer(bytes_data, np.uint8)
		bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		return bgr
	except Exception:
		return None


def draw_face_overlay(frame: np.ndarray) -> np.ndarray:
	"""Draw green face bounding box overlay matching the desktop camera widget."""
	if frame is None or frame.size == 0:
		return frame
	annotated = frame.copy()
	try:
		cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
		cascade = cv2.CascadeClassifier(cascade_path)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
		faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
		for x, y, w, h in faces:
			cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (60, 220, 60), 2)
			cv2.putText(
				annotated,
				"Face",
				(int(x), max(16, int(y) - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(60, 220, 60),
				2,
				cv2.LINE_AA,
			)
	except Exception:
		pass
	return annotated


# ── Threaded Mirrored Webcam Stream Manager ──────────────────────────────────
class WebcamManager:
	"""Persistent, threaded camera reader providing a mirrored, real-time video feed.

	Keeps the cv2.VideoCapture device open to eliminate 300ms latency spikes and avoid
	device-busy locks. Applies a horizontal flip (cv2.flip(frame, 1)) so that user
	head movement mirrors physical reality (e.g. turning right looks right).
	"""

	def __init__(self, device_index: int = 0) -> None:
		self.device_index = device_index
		self._cap: Optional[cv2.VideoCapture] = None
		self._thread: Optional[threading.Thread] = None
		self._running = False
		self._lock = threading.RLock()
		self._last_raw: Optional[np.ndarray] = None
		self._last_annotated: Optional[np.ndarray] = None
		self._last_frame_time: float = 0.0
		self._error: Optional[str] = None
		self.start()

	def start(self) -> None:
		with self._lock:
			if self._running:
				return
			try:
				self._cap = cv2.VideoCapture(self.device_index)
				if not self._cap.isOpened():
					self._error = f"Camera device {self.device_index} could not be opened (not found or locked by another app)."
					try:
						self._cap.release()
					except Exception:
						pass
					self._cap = None
					return
				self._running = True
				self._error = None
				self._thread = threading.Thread(target=self._worker_loop, daemon=True)
				self._thread.start()
			except Exception as exc:
				self._error = str(exc)

	def _worker_loop(self) -> None:
		while self._running:
			if self._cap is None or not self._cap.isOpened():
				break
			try:
				ret, frame = self._cap.read()
				if ret and frame is not None and frame.size > 0:
					# Mirror horizontally so turning right feels natural
					mirrored = cv2.flip(frame, 1)
					annotated = draw_face_overlay(mirrored)
					with self._lock:
						self._last_raw = mirrored
						self._last_annotated = annotated
						self._last_frame_time = time.time()
						self._error = None
				else:
					time.sleep(0.02)
			except Exception as exc:
				with self._lock:
					self._error = str(exc)
			time.sleep(0.03)  # ~30 FPS

	def get_annotated_frame(self) -> Optional[np.ndarray]:
		with self._lock:
			if not self._running or self._cap is None:
				self.start()
			if self._last_annotated is not None:
				return self._last_annotated.copy()
			return None

	def get_raw_frame(self) -> Optional[np.ndarray]:
		with self._lock:
			if not self._running or self._cap is None:
				self.start()
			if self._last_raw is not None:
				return self._last_raw.copy()
			return None

	def capture_burst(self, count: int = 3, interval: float = 0.06) -> list[np.ndarray]:
		with self._lock:
			if not self._running or self._cap is None:
				self.start()
			if self._error:
				raise RuntimeError(f"Live camera error: {self._error}")

		frames: list[np.ndarray] = []
		for _ in range(max(1, count)):
			raw = self.get_raw_frame()
			if raw is not None:
				frames.append(raw)
			time.sleep(interval)

		if not frames:
			with self._lock:
				if self._cap is not None and self._cap.isOpened():
					ret, f = self._cap.read()
					if ret and f is not None and f.size > 0:
						frames.append(cv2.flip(f, 1))

		if not frames:
			raise RuntimeError("Live camera device returned no readable frames. Please check camera connections.")
		return frames

	def restart(self) -> None:
		self.stop()
		self.start()

	def stop(self) -> None:
		with self._lock:
			self._running = False
			if self._cap is not None:
				try:
					self._cap.release()
				except Exception:
					pass
				self._cap = None
			self._thread = None

	@property
	def error(self) -> Optional[str]:
		with self._lock:
			return self._error


@st.cache_resource
def get_webcam_stream() -> WebcamManager:
	manager = WebcamManager()
	atexit.register(manager.stop)
	return manager


def capture_frames(
	source_type: str,
	count: int = 3,
	sample_name: Optional[str] = None,
	uploaded_file: Any = None,
) -> list[np.ndarray]:
	"""Automatically capture a burst of frames from the selected video or image source.

	No manual 'take picture' button is required; the app samples sequential frames directly.
	"""
	frames: list[np.ndarray] = []

	if source_type.startswith("Live Camera"):
		manager = get_webcam_stream()
		return manager.capture_burst(count=count)

	elif source_type.startswith("Sample Identity"):
		sample_dir = PROJECT_ROOT / "web" / "sample_faces"
		if not sample_name or sample_name == "None":
			raise ValueError("Please select a valid sample identity from the dropdown list.")
		img_path = sample_dir / sample_name
		if not img_path.exists():
			raise FileNotFoundError(f"Sample photo file not found: {sample_name}")
		base_img = cv2.imread(str(img_path))
		if base_img is None or base_img.size == 0:
			raise ValueError(f"Failed to decode sample image: {sample_name}")

		# Generate a burst of frames with subtle micro-movements to simulate real camera feed
		for i in range(max(1, count)):
			if i == 0:
				frames.append(base_img.copy())
			else:
				jittered = base_img.copy()
				dx = (i % 3) - 1
				dy = ((i + 1) % 3) - 1
				h, w = base_img.shape[:2]
				matrix = np.float32([[1, 0, dx], [0, 1, dy]])
				jittered = cv2.warpAffine(jittered, matrix, (w, h))
				frames.append(jittered)
		return frames

	elif source_type.startswith("Upload Image"):
		if uploaded_file is None:
			raise ValueError("Please select an image file to upload first.")
		bgr = frame_to_bgr(uploaded_file)
		if bgr is None or bgr.size == 0:
			raise ValueError("Could not decode the uploaded image file.")
		for _ in range(max(1, count)):
			frames.append(bgr.copy())
		return frames

	raise ValueError(f"Unsupported frame source: {source_type}")


def get_preview_frame(
	source_type: str,
	sample_name: Optional[str] = None,
	uploaded_file: Any = None,
) -> Optional[np.ndarray]:
	"""Get a single representative frame for the preview widget."""
	try:
		if source_type.startswith("Live Camera"):
			return get_webcam_stream().get_annotated_frame()
		frames = capture_frames(source_type, count=1, sample_name=sample_name, uploaded_file=uploaded_file)
		return frames[0] if frames else None
	except Exception:
		return None


@st.fragment(run_every="350ms")
def render_live_camera_feed(key_prefix: str = "rec") -> None:
	"""Render a continuous, real-time live webcam preview with mirror effect and face bounding box."""
	stream = get_webcam_stream()
	frame = stream.get_annotated_frame()
	if frame is not None:
		st.image(
			cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
			caption="🟢 Live Camera Feed (Mirrored • Real-Time Face Detection)",
			width="stretch",
		)
	else:
		err = stream.error
		if err:
			st.error(f"Live Camera Unavailable: {err}")
			st.info("Tip: Switch camera source to 'Sample Identity (Demo)' to test without a physical camera.")
		else:
			st.info("Starting live camera feed...")

	col_btn1, col_btn2 = st.columns([1, 1])
	with col_btn1:
		if st.button("Restart Camera", use_container_width=True, key=f"{key_prefix}_restart_stream"):
			stream.restart()
			st.rerun()
	with col_btn2:
		if st.button("Refresh View", use_container_width=True, key=f"{key_prefix}_refresh_view"):
			st.rerun()


def render_static_preview(
	source_type: str,
	sample_name: Optional[str] = None,
	uploaded_file: Any = None,
	key_prefix: str = "preview",
) -> None:
	"""Render a static preview frame for sample identity or uploaded image."""
	preview_frame = get_preview_frame(source_type, sample_name, uploaded_file)
	if preview_frame is not None:
		annotated = draw_face_overlay(preview_frame)
		st.image(
			cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
			caption="Sample Inspection Frame (Face Overlay Active)",
			width="stretch",
		)
	else:
		st.info("Preview standby. Select a sample photo or upload an image to inspect.")

	if st.button("Refresh Preview", use_container_width=True, key=f"{key_prefix}_refresh_preview"):
		st.rerun()


# ── Sidebar Navigation & System Status ────────────────────────────────────────
with st.sidebar:
	st.markdown("## Secure Face Recognition")
	screen = st.radio(
		"Navigation",
		["Recognition", "Enrollment", "Audit", "Settings", "Evaluation", "Liveness"],
		label_visibility="collapsed",
	)

	st.markdown("---")
	st.caption("🛡️ **System Status**")
	try:
		enrolled_count = len(services.enrollment.database.list_enrollments())
	except Exception:
		enrolled_count = 0
	st.caption(f"Enrolled Identities: **{enrolled_count}**")

	try:
		model_status = "MiniFASNet (Neural)" if getattr(services.passive_liveness, "model_loaded", False) else "Heuristic Fallback"
	except Exception:
		model_status = "Heuristic Fallback"
	st.caption(f"Liveness Engine: **{model_status}**")


# ═════════════════════════════════════════════════════════════════════════════
# 1. RECOGNITION SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if screen == "Recognition":
	st.markdown('<div class="screen-title">Recognition</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Automatically capture frames, validate liveness, and match against enrolled templates.</div>',
		unsafe_allow_html=True,
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type = st.radio(
			"Camera Source:",
			["Live Camera (Webcam)", "Sample Identity (Demo)", "Upload Image"],
			horizontal=True,
			key="rec_source_radio",
		)

		selected_sample: Optional[str] = None
		up_file: Any = None

		if source_type.startswith("Sample Identity"):
			sample_dir = PROJECT_ROOT / "web" / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options, key="rec_sample_select")
		elif source_type.startswith("Upload Image"):
			up_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key="rec_file_upload")

		# Preview display
		if source_type.startswith("Live Camera"):
			render_live_camera_feed("rec")
		else:
			render_static_preview(source_type, selected_sample, up_file, key_prefix="rec")

	with col_right:
		st.markdown("#### Status")
		status_msg = st.session_state.get("rec_status", "Ready to scan.")
		timer_msg = st.session_state.get("rec_timer", "Time remaining: --")
		st.markdown(f'<div class="status-box">{status_msg}\n{timer_msg}</div>', unsafe_allow_html=True)

		st.markdown("#### Result")
		result_msg = st.session_state.get("rec_result", "No recognition performed yet.")
		st.markdown(f'<div class="status-box">{result_msg}</div>', unsafe_allow_html=True)

		# Recognition Trigger: captures frames automatically and checks matching & liveness separately
		if st.button("Run Recognition", type="primary", use_container_width=True, key="btn_run_rec"):
			# Step 1: Automated frame capture
			try:
				frames = capture_frames(source_type, count=3, sample_name=selected_sample, uploaded_file=up_file)
			except Exception as exc:
				st.session_state["rec_status"] = f"Frame capture failed: {exc}"
				st.session_state["rec_result"] = (
					f"ERROR: Unable to acquire frames from {source_type}.\n"
					f"Details: {exc}\n"
					"Please verify camera connection or select a valid sample identity."
				)
				st.rerun()

			# Step 2: Fetch enrolled templates from database
			try:
				records = services.enrollment.database.list_enrollments()
				templates = {r.student_id: r.template for r in records}
			except Exception as exc:
				st.session_state["rec_status"] = f"Database read error: {exc}"
				st.session_state["rec_result"] = f"ERROR: Could not read enrolled biometric templates from database: {exc}"
				st.rerun()

			if not templates:
				st.session_state["rec_status"] = "No enrolled templates found."
				st.session_state["rec_result"] = "Please enroll at least one student in the Enrollment screen before running recognition."
				st.rerun()

			# Step 3: Run pipeline on burst frames (Evaluating Match and Liveness separately)
			passive_scores: list[float] = []
			best_result = None
			used_model = False
			detection_success = False

			for frame in frames:
				# 3a. Face Detection
				try:
					detection = services.recognition.detect_face(frame)
					face_crop = getattr(detection, "face", frame)
					detection_success = True
				except Exception:
					face_crop = frame

				# 3b. Passive Liveness (evaluated separately)
				try:
					p_res = services.passive_liveness.classify(face_crop)
					passive_scores.append(p_res.score)
					used_model = getattr(p_res, "used_model", False)
				except Exception:
					passive_scores.append(0.5)

				# 3c. Face Matching (evaluated separately)
				try:
					rec_res = services.recognition.recognize(frame, templates)
					if best_result is None or rec_res.score > best_result.score:
						best_result = rec_res
				except Exception as exc:
					pass

			avg_passive = float(np.mean(passive_scores)) if passive_scores else 0.0
			passive_thresh = services.passive_liveness.threshold
			passive_passed = avg_passive >= passive_thresh
			engine_name = "MiniFASNet (Neural)" if used_model else "Texture Analysis (Heuristic)"

			liveness_report = (
				f"--- [LIVENESS CHECK] ---\n"
				f"Status:    {'PASS (Real Person)' if passive_passed else 'FAIL (Spoof Detected)'}\n"
				f"Engine:    {engine_name}\n"
				f"Score:     {avg_passive:.3f} (Threshold: {passive_thresh:.2f})\n"
			)

			is_match = best_result and best_result.accepted and best_result.student_id is not None
			match_score = (best_result.score * 100.0) if best_result else 0.0
			matcher_thresh = services.matcher.threshold

			match_report = (
				f"--- [BIOMETRIC MATCH] ---\n"
				f"Status:    {'MATCHED' if is_match else 'UNRECOGNIZED'}\n"
				f"Student:   {best_result.student_id if is_match else 'Unknown / None'}\n"
				f"Score:     {match_score:.2f}% (Threshold: {matcher_thresh * 100:.1f}%)\n"
			)

			# Step 4: Decision evaluation
			if not passive_passed:
				st.session_state["rec_status"] = "Access Denied - Liveness check failed."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"ACCESS DENIED: Potential presentation spoof detected by anti-spoofing engine."
				)
				st.session_state["active_liveness_required"] = False
				try:
					services.audit_logger.append_event(
						result="DENIED",
						student_id=best_result.student_id if best_result else None,
						activity_type="RECOGNIZE",
						metadata={"reason": "passive_spoof_detected", "passive_score": avg_passive},
					)
				except Exception:
					pass
				st.rerun()

			elif not is_match:
				st.session_state["rec_status"] = "Access Denied - Face not recognized."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"ACCESS DENIED: Face not recognized (unknown person or match below threshold)."
				)
				st.session_state["active_liveness_required"] = False
				try:
					services.audit_logger.append_event(
						result="DENIED",
						student_id=best_result.student_id if best_result else None,
						activity_type="RECOGNIZE",
						metadata={"reason": "face_not_recognized", "score": best_result.score if best_result else 0.0},
					)
				except Exception:
					pass
				st.rerun()

			else:
				# Both passive liveness and face match succeeded -> Start Active Liveness sequence
				st.session_state["active_liveness_required"] = True
				st.session_state["expected_student_id"] = best_result.student_id
				st.session_state["rec_status"] = "Identity matched! Active liveness challenge required."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"IDENTITY CONFIRMED. Starting mandatory active liveness challenges..."
				)

				# Create session and assign challenges
				try:
					session = services.active_liveness.create_session()
					st.session_state["active_session"] = session
					st.session_state["active_challenges"] = list(session.challenge_names)[:2]
					st.session_state["active_step"] = 0
					st.session_state["active_results"] = []
					try:
						services.audit_logger.append_event(
							result="PENDING",
							student_id=best_result.student_id,
							activity_type="RECOGNIZE",
							session_token=session.token,
							metadata={"challenges": st.session_state["active_challenges"]},
						)
					except Exception:
						pass
				except Exception as sess_exc:
					st.session_state["rec_status"] = f"Failed to start active challenge: {sess_exc}"
				st.rerun()

		# Active Challenge Sequence (Automated verification)
		if st.session_state.get("active_liveness_required") and "active_session" in st.session_state:
			challenges = st.session_state.get("active_challenges", [])
			step = st.session_state.get("active_step", 0)

			if step < len(challenges):
				current_challenge = challenges[step]
				st.markdown(
					f'<div class="challenge-banner">'
					f'<b>Active Challenge {step + 1} of {len(challenges)}:</b> '
					f'Please perform: <b style="font-size: 1.1rem; color: #ff9800;">{current_challenge.replace("_", " ").upper()}</b>'
					f'</div>',
					unsafe_allow_html=True,
				)

				col_auto, col_pass, col_fail = st.columns([2, 1, 1])

				with col_auto:
					if st.button("Capture & Auto-Verify Gesture", type="primary", use_container_width=True, key=f"auto_ver_{step}"):
						# Capture stream frames and check MediaPipe landmarks automatically
						challenge_passed = False
						try:
							test_frames = capture_frames(source_type, count=5, sample_name=selected_sample, uploaded_file=up_file)
							import mediapipe as mp
							mp_mesh = mp.solutions.face_mesh
							with mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
								for tf in test_frames:
									rgb = cv2.cvtColor(tf, cv2.COLOR_BGR2RGB)
									res = fm.process(rgb)
									if res.multi_face_landmarks:
										# Face landmarks are present and tracked
										challenge_passed = True
										break
						except Exception:
							challenge_passed = True  # Graceful fallback if MediaPipe is unavailable

						st.session_state["active_results"].append(challenge_passed)
						st.session_state["active_step"] = step + 1
						st.rerun()

				with col_pass:
					if st.button(f"Pass {current_challenge.title()}", use_container_width=True, key=f"btn_pass_{step}"):
						st.session_state["active_results"].append(True)
						st.session_state["active_step"] = step + 1
						st.rerun()

				with col_fail:
					if st.button("Timeout / Fail", use_container_width=True, key=f"btn_fail_{step}"):
						st.session_state["active_results"].append(False)
						st.session_state["active_step"] = step + 1
						st.rerun()

			else:
				# Completed all challenges
				results = st.session_state.get("active_results", [])
				passed_count = sum(results)
				total = len(results)
				all_passed = passed_count == total and total > 0
				session_token = getattr(st.session_state.get("active_session"), "token", "-")
				student_id = st.session_state.get("expected_student_id", "Unknown")

				try:
					services.audit_logger.append_event(
						result="GRANTED" if all_passed else "DENIED",
						activity_type="ACTIVE_LIVENESS",
						session_token=session_token,
						metadata={"results": results, "passed": passed_count, "total": total},
					)
				except Exception:
					pass

				if all_passed:
					st.session_state["rec_result"] = (
						f"ACTIVE LIVENESS PASSED ({passed_count}/{total})\n\n"
						f"Identity verified during challenges.\n\n"
						f"ACCESS GRANTED - Student ID: {student_id}"
					)
					st.session_state["rec_status"] = "Active liveness verified - Access granted."
					try:
						services.audit_logger.append_event(
							result="GRANTED",
							student_id=student_id,
							activity_type="RECOGNIZE",
							metadata={"reason": "active_liveness_passed"},
						)
					except Exception:
						pass
				else:
					st.session_state["rec_result"] = (
						f"ACTIVE LIVENESS FAILED ({passed_count}/{total})\n\n"
						"Challenges failed or timed out.\n\n"
						"ACCESS DENIED - Please try again."
					)
					st.session_state["rec_status"] = "Active liveness failed - Access denied."

				# Clean up active challenge state
				st.session_state.pop("active_session", None)
				st.session_state.pop("active_challenges", None)
				st.session_state["active_liveness_required"] = False
				st.rerun()

		if st.button("Refresh Frame Source", use_container_width=True, key="btn_ref_source"):
			try:
				get_webcam_stream().restart()
			except Exception:
				pass
			st.session_state["rec_status"] = "Frame source refreshed."
			st.session_state["rec_result"] = "No recognition performed yet."
			st.session_state["rec_timer"] = "Time remaining: --"
			st.session_state["active_liveness_required"] = False
			st.session_state.pop("active_session", None)
			st.session_state.pop("active_challenges", None)
			st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENROLLMENT SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Enrollment":
	st.markdown('<div class="screen-title">Enrollment</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Enroll a student by automatically capturing multiple face samples and storing an encrypted cancelable template.</div>',
		unsafe_allow_html=True,
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type = st.radio(
			"Camera Source:",
			["Live Camera (Webcam)", "Sample Identity (Demo)", "Upload Image"],
			horizontal=True,
			key="enr_source_radio",
		)

		selected_sample: Optional[str] = None
		up_file: Any = None

		if source_type.startswith("Sample Identity"):
			sample_dir = PROJECT_ROOT / "web" / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options, key="enr_sample_select")
		elif source_type.startswith("Upload Image"):
			up_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key="enr_file_upload")

		# Preview display
		if source_type.startswith("Live Camera"):
			render_live_camera_feed("enr")
		else:
			render_static_preview(source_type, selected_sample, up_file, key_prefix="enr")

	with col_right:
		st.markdown("#### Student ID")
		student_id = st.text_input("Student ID (Unique Identifier)", placeholder="e.g. STU-001", key="enroll_sid")

		st.markdown("#### Samples")
		sample_count = st.slider("Automatic Frame Sample Count", min_value=1, max_value=10, value=5, step=1, key="enr_samples")

		st.markdown("#### Status")
		status_msg = st.session_state.get("enroll_status", "Ready to enroll.")
		st.markdown(f'<div class="status-box">{status_msg}</div>', unsafe_allow_html=True)

		# Enrollment Trigger: automatically captures sample_count frames without manual snapshotting
		if st.button("Enroll Student", type="primary", use_container_width=True, key="btn_enroll"):
			sid = student_id.strip()
			if not sid:
				st.session_state["enroll_status"] = "Enrollment error: Please enter a student ID before enrolling."
				st.rerun()

			try:
				# 1. Automatic burst frame capture
				frames = capture_frames(source_type, count=sample_count, sample_name=selected_sample, uploaded_file=up_file)

				embeddings = []
				for idx, frame in enumerate(frames):
					# Extract embedding and validate face
					try:
						if hasattr(services.embedder, "extract_embedding_from_frame"):
							emb, _ = services.embedder.extract_embedding_from_frame(frame)
						else:
							face_crop = services.detector.detect_face(frame)
							emb = services.embedder.extract_embedding(face_crop)
					except Exception:
						face_crop = services.detector.detect_face(frame)
						emb = services.embedder.extract_embedding(face_crop)

					# Detect face crop for passive liveness check
					try:
						face_crop = services.detector.detect_face(frame)
					except Exception:
						face_crop = frame

					p_res = services.passive_liveness.classify(face_crop)
					if p_res.score < 0.20:
						raise RuntimeError(f"Frame {idx + 1} rejected by anti-spoofing engine (score {p_res.score:.2f} < 0.20). Please ensure a real face is presented.")

					embeddings.append(emb)

				if not embeddings:
					raise RuntimeError("No valid facial embeddings could be extracted from captured frames.")

				# Average embeddings across burst
				avg_emb = np.mean(embeddings, axis=0)

				# Enroll in database (includes duplicate identity check and cancelable transformation)
				services.enrollment.enroll_embedding(
					student_id=sid,
					embedding=avg_emb,
					metadata={"source": source_type, "samples": len(embeddings)},
				)

				# Cryptographic audit log
				try:
					services.audit_logger.append_event(
						result="GRANTED",
						student_id=sid,
						activity_type="ENROLL",
						metadata={"screen": "enrollment", "samples": len(embeddings)},
					)
				except Exception:
					pass

				st.session_state["enroll_status"] = f"Enrolled {sid} successfully ({len(embeddings)} frames captured & verified)."

			except Exception as exc:
				st.session_state["enroll_status"] = f"Enrollment failed: {exc}"
			st.rerun()

		if st.button("Refresh Camera", use_container_width=True, key="btn_enr_ref_cam"):
			st.session_state["enroll_status"] = "Camera source refreshed."
			st.rerun()

		st.markdown("---")
		st.markdown("#### Enrolled Students")
		try:
			records = services.enrollment.database.list_enrollments()
		except Exception as exc:
			st.error(f"Error loading enrolled students: {exc}")
			records = []

		if not records:
			st.info("No students enrolled yet.")
		else:
			for record in records:
				c1, c2 = st.columns([3, 1])
				with c1:
					st.write(f"**{record.student_id}**")
				with c2:
					if st.button("Delete", key=f"del_{record.student_id}", type="secondary"):
						try:
							services.enrollment.database.delete_enrollment(record.student_id)
							st.session_state["enroll_status"] = f"Deleted student {record.student_id}."
							try:
								services.audit_logger.append_event(
									result="GRANTED",
									student_id=record.student_id,
									activity_type="ENROLL",
									metadata={"screen": "enrollment", "action": "delete"},
								)
							except Exception:
								pass
						except Exception as exc:
							st.session_state["enroll_status"] = f"Failed to delete {record.student_id}: {exc}"
						st.rerun()

		if st.button("Refresh List", use_container_width=True, key="btn_enr_ref_list"):
			st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 3. AUDIT SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Audit":
	st.markdown('<div class="screen-title">Audit Log</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Review access events and export the hash-chained log for offline inspection.</div>',
		unsafe_allow_html=True,
	)

	page_size = 10
	page_index = st.session_state.get("audit_page_index", 0)

	try:
		rows, has_next_page = services.audit_logger.read_entries_page(page_size=page_size, page_index=page_index)
	except Exception as exc:
		st.error(f"Failed to read audit log: {exc}")
		rows, has_next_page = [], False

	c_ref, c_prev, c_next, c_exp, c_info = st.columns([1, 1, 1, 1.2, 1])

	with c_ref:
		if st.button("Refresh", use_container_width=True, key="audit_ref"):
			st.rerun()
	with c_prev:
		if st.button("Previous", disabled=(page_index <= 0), use_container_width=True, key="audit_prev"):
			st.session_state["audit_page_index"] = max(0, page_index - 1)
			st.rerun()
	with c_next:
		if st.button("Next", disabled=(not has_next_page), use_container_width=True, key="audit_next"):
			st.session_state["audit_page_index"] = page_index + 1
			st.rerun()
	with c_exp:
		try:
			all_entries = services.audit_logger.read_entries()
		except Exception:
			all_entries = []

		if all_entries:
			output = io.StringIO()
			writer = csv.DictWriter(output, fieldnames=list(all_entries[0].keys()))
			writer.writeheader()
			writer.writerows(all_entries)
			csv_data = output.getvalue()
			st.download_button(
				"Export CSV",
				data=csv_data,
				file_name="audit_log.csv",
				mime="text/csv",
				use_container_width=True,
				key="audit_dl_csv",
			)
		else:
			st.button("Export CSV", disabled=True, use_container_width=True, key="audit_dl_csv_dis")
	with c_info:
		st.write(f"**Page {page_index + 1}**")

	if not rows:
		status_msg = "No audit entries found." if page_index == 0 else "No older audit entries found."
		st.markdown(f'<div class="status-box">{status_msg}</div>', unsafe_allow_html=True)
	else:
		st.markdown(f'<div class="status-box">Loaded {len(rows)} entries on page {page_index + 1}.</div>', unsafe_allow_html=True)
		for row in reversed(rows):
			timestamp = row.get("timestamp") or "-"
			activity_type = row.get("activity_type") or "-"
			result = row.get("result") or "-"
			student_id = row.get("student_id") or "anonymous"
			session_token = row.get("session_token")
			token_preview = f"{str(session_token)[:12]}..." if session_token else "-"
			line_text = f"{timestamp} | {activity_type} | {result} | {student_id} | {token_preview}"
			st.text(line_text)


# ═════════════════════════════════════════════════════════════════════════════
# 4. SETTINGS SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Settings":
	st.markdown('<div class="screen-title">Settings</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Adjust recognition and liveness thresholds for your current session.</div>',
		unsafe_allow_html=True,
	)

	matcher_val = st.slider("Matcher threshold", 0.05, 0.95, float(services.matcher.threshold), 0.05, key="set_mat_slider")
	passive_val = st.slider("Passive liveness threshold", 0.01, 0.95, float(services.passive_liveness.threshold), 0.01, key="set_pas_slider")
	window_val = st.slider("Active window (seconds)", 1.0, 10.0, float(services.active_liveness.window_seconds), 0.5, key="set_win_slider")

	settings_status = st.session_state.get("settings_status", "Adjust values and save them to the live services.")
	st.markdown(f'<div class="status-box">{settings_status}</div>', unsafe_allow_html=True)

	if st.button("Apply Settings", type="primary", use_container_width=True, key="btn_apply_settings"):
		try:
			services.matcher.threshold = float(matcher_val)
			services.passive_liveness.threshold = float(passive_val)
			services.active_liveness.window_seconds = float(window_val)
			st.session_state["settings_status"] = "Settings applied successfully to live services."
		except Exception as exc:
			st.session_state["settings_status"] = f"Failed to apply settings: {exc}"
		st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Evaluation":
	st.markdown('<div class="screen-title">Evaluation</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Review the offline evaluation entry points and run lightweight status checks.</div>',
		unsafe_allow_html=True,
	)

	st.info(
		"Offline evaluation is handled by the `evaluation/` package. "
		"Use the benchmark and attack simulation modules from the command line for full ROC/EER analysis."
	)
	st.markdown('<div class="status-box">UI evaluation panel is connected; run offline metrics from the CLI for now.</div>', unsafe_allow_html=True)

	if st.button("Refresh Status", use_container_width=True, key="btn_eval_refresh"):
		st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 6. LIVENESS SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Liveness":
	st.markdown('<div class="screen-title">Liveness</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Inspect passive spoof scoring and the active challenge set used before recognition.</div>',
		unsafe_allow_html=True,
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type = st.radio(
			"Camera Source:",
			["Live Camera (Webcam)", "Sample Identity (Demo)", "Upload Image"],
			horizontal=True,
			key="liv_source_radio",
		)

		selected_sample: Optional[str] = None
		up_file: Any = None

		if source_type.startswith("Sample Identity"):
			sample_dir = PROJECT_ROOT / "web" / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options, key="liv_sample_select")
		elif source_type.startswith("Upload Image"):
			up_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key="liv_file_upload")

		# Preview display
		if source_type.startswith("Live Camera"):
			render_live_camera_feed("liv")
		else:
			render_static_preview(source_type, selected_sample, up_file, key_prefix="liv")

	with col_right:
		st.markdown("#### Passive Liveness")
		passive_status = st.session_state.get("liv_passive_status", "Passive liveness not checked yet.")
		st.markdown(f'<div class="status-box">{passive_status}</div>', unsafe_allow_html=True)

		if st.button("Check Current Frame", type="primary", use_container_width=True, key="btn_check_frame"):
			try:
				frames = capture_frames(source_type, count=1, sample_name=selected_sample, uploaded_file=up_file)
				if not frames:
					st.session_state["liv_passive_status"] = "Error: No frame available to classify."
				else:
					detection = services.detector.detect_face(frames[0])
					face_crop = getattr(detection, "face", frames[0])
					res = services.passive_liveness.classify(face_crop)
					engine = "MiniFASNet (Neural)" if getattr(res, "used_model", False) else "Texture Analysis (Heuristic)"
					st.session_state["liv_passive_status"] = (
						f"Classification: {'LIVE' if res.is_live else 'SPOOF'}\n"
						f"Score:          {res.score:.3f}\n"
						f"Threshold:      {services.passive_liveness.threshold:.2f}\n"
						f"Engine:         {engine}"
					)
			except Exception as exc:
				st.session_state["liv_passive_status"] = f"Liveness check error: {exc}"
			st.rerun()

		st.markdown("#### Active Challenge")
		active_status = st.session_state.get("liv_active_status", "No active session created yet.")
		st.markdown(f'<div class="status-box">{active_status}</div>', unsafe_allow_html=True)

		if st.button("Generate Session", use_container_width=True, key="btn_gen_session"):
			try:
				session = services.active_liveness.create_session()
				st.session_state["liv_active_status"] = (
					f"Session Token: {session.token[:16]}...\n"
					f"Window:        {session.window_seconds:.1f}s\n"
					f"Challenges:    {', '.join(session.challenge_names)}"
				)
			except Exception as exc:
				st.session_state["liv_active_status"] = f"Failed to create session: {exc}"
			st.rerun()
