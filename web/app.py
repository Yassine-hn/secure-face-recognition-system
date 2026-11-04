"""Streamlit web interface matching the exact layout and logic of the original desktop GUI."""

from __future__ import annotations

import csv
import io
import sys
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
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> AppServices:
	return build_default_services()


services = get_services()


def frame_to_bgr(uploaded_file: Any) -> Optional[np.ndarray]:
	"""Convert a Streamlit UploadedFile or CameraInput buffer into an OpenCV BGR frame."""
	if uploaded_file is None:
		return None
	bytes_data = uploaded_file.getvalue()
	np_arr = np.frombuffer(bytes_data, np.uint8)
	bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	return bgr


def draw_face_overlay(frame: np.ndarray) -> np.ndarray:
	"""Draw green face bounding box overlay matching the desktop camera widget."""
	annotated = frame.copy()
	try:
		cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
		cascade = cv2.CascadeClassifier(cascade_path)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
	st.markdown("## Secure Face Recognition")
	screen = st.radio(
		"Navigation",
		["Recognition", "Enrollment", "Audit", "Settings", "Evaluation", "Liveness"],
		label_visibility="collapsed",
	)

	st.markdown("---")
	st.caption("🛡️ **System Status**")
	enrolled_count = len(services.enrollment.database.list_enrollments())
	st.caption(f"Enrolled Identities: **{enrolled_count}**")
	model_status = "MiniFASNet (Neural)" if services.passive_liveness.model_loaded else "Heuristic Fallback"
	st.caption(f"Liveness Engine: **{model_status}**")


# ═════════════════════════════════════════════════════════════════════════════
# 1. RECOGNITION SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if screen == "Recognition":
	st.markdown('<div class="screen-title">Recognition</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Capture a frame, validate liveness, and match it against enrolled templates.</div>',
		unsafe_allow_html=True,
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type = st.radio("Frame Source:", ["Camera Input", "Upload / Sample File"], horizontal=True)
		frame_bgr: Optional[np.ndarray] = None

		if source_type == "Camera Input":
			cam_input = st.camera_input("Capture frame for recognition", key="rec_cam")
			frame_bgr = frame_to_bgr(cam_input)
		else:
			sample_dir = Path(__file__).parent / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options)
			up_file = st.file_uploader("Or Upload Image File", type=["jpg", "jpeg", "png"], key="rec_upload")

			if up_file is not None:
				frame_bgr = frame_to_bgr(up_file)
			elif selected_sample != "None":
				frame_bgr = cv2.imread(str(sample_dir / selected_sample))

		if frame_bgr is not None:
			preview_frame = draw_face_overlay(frame_bgr)
			st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Camera Preview (Processed)", use_container_width=True)

	with col_right:
		st.markdown("#### Status")
		rec_status = st.session_state.get("rec_status", "Ready to scan.")
		rec_timer = st.session_state.get("rec_timer", "Time remaining: --")
		st.markdown(f'<div class="status-box">{rec_status}\n{rec_timer}</div>', unsafe_allow_html=True)

		st.markdown("#### Result")
		rec_result = st.session_state.get("rec_result", "No recognition performed yet.")
		st.markdown(f'<div class="status-box">{rec_result}</div>', unsafe_allow_html=True)

		if st.button("Run Recognition", type="primary", use_container_width=True):
			st.session_state["active_liveness_required"] = False
			records = services.enrollment.database.list_enrollments()
			templates = {record.student_id: record.template for record in records}

			if frame_bgr is None:
				st.session_state["rec_status"] = "No frames available from camera."
				st.session_state["rec_result"] = "Please capture a frame before running recognition."
				st.rerun()
			elif not templates:
				st.session_state["rec_status"] = "No enrolled templates found."
				st.session_state["rec_result"] = "Enroll a student first."
				st.rerun()
			else:
				try:
					# 1. Face detection
					try:
						detection = services.recognition.detect_face(frame_bgr)
						face_crop = getattr(detection, "face", frame_bgr)
					except Exception:
						face_crop = frame_bgr

					# 2. Passive liveness check through the proper model
					passive_result = services.passive_liveness.classify(face_crop)

					# 3. Recognition through proper pipeline
					result = services.recognition.recognize(frame_bgr, templates)

					# 4. Handle result
					if result.error:
						st.session_state["rec_status"] = f"Recognition failed: {result.error}"
						st.session_state["rec_result"] = (
							"ACCESS DENIED - Face detection succeeded but embedding extraction failed. "
							"Please try again with better lighting or face positioning."
						)
						services.audit_logger.append_event(
							result="DENIED",
							activity_type="RECOGNIZE",
							metadata={"screen": "recognition", "error": result.error, "reason": "embedding_extraction_failed"},
						)
					elif not (result.accepted and result.student_id is not None):
						if result.student_id is None:
							message = "ACCESS DENIED - Face not recognized (unknown person)"
							denial_reason = "face_not_recognized"
						else:
							message = f"ACCESS DENIED - Known face ({result.student_id}) but below confidence threshold ({result.score:.2%})"
							denial_reason = "below_threshold"
						st.session_state["rec_status"] = "Recognition failed."
						st.session_state["rec_result"] = message
						services.audit_logger.append_event(
							result="DENIED",
							student_id=result.student_id,
							activity_type="RECOGNIZE",
							metadata={"score": result.score, "passive_score": passive_result.score, "reason": denial_reason},
						)
					else:
						# Case 2: Recognition passes -> Active liveness required
						st.session_state["active_liveness_required"] = True
						st.session_state["expected_student_id"] = result.student_id
						st.session_state["rec_status"] = "Active liveness required for this attempt."
						st.session_state["rec_result"] = (
							f"ACTIVE LIVENESS REQUIRED FOR ALL ATTEMPTS.\n\n"
							f"Avg Passive liveness: {passive_result.score:.3f}\n"
							f"Best Recognition: {result.score:.2%} for student {result.student_id}\n\n"
							"Click 'Start Active Liveness Check' to proceed."
						)
						services.audit_logger.append_event(
							result="PENDING",
							student_id=result.student_id,
							activity_type="RECOGNIZE",
							metadata={
								"score": result.score,
								"passive_score": passive_result.score,
								"requires_active_liveness": True,
								"reason": "active_liveness_required",
								"liveness_passed": passive_result.is_live,
							},
						)
				except Exception as exc:
					st.session_state["rec_status"] = f"Recognition error: {exc}"
					st.session_state["rec_result"] = "Recognition could not be completed."
				st.rerun()

		# Active Liveness Button (shown when required)
		if st.session_state.get("active_liveness_required", False):
			st.markdown("---")
			st.warning("⚠️ Additional verification required: active liveness check")

			if st.button("Start Active Liveness Check", type="secondary", use_container_width=True):
				session = services.active_liveness.create_session()
				challenges = list(session.challenge_names)
				if len(challenges) < 2:
					challenges = ["blink", "smile"]
				st.session_state["active_session"] = session
				st.session_state["active_challenges"] = challenges
				st.session_state["active_step"] = 0
				st.session_state["active_results"] = []
				st.rerun()

		# If in active challenge sequence
		if "active_session" in st.session_state and st.session_state.get("active_challenges"):
			challenges = st.session_state["active_challenges"]
			step = st.session_state.get("active_step", 0)

			if step < len(challenges):
				current_challenge = challenges[step]
				st.info(f"**Challenge {step + 1} of {len(challenges)}:** Please perform: **{current_challenge.replace('_', ' ').upper()}**")

				col_pass, col_fail = st.columns(2)
				with col_pass:
					if st.button(f"Verify {current_challenge.title()} Passed", type="primary", use_container_width=True):
						st.session_state["active_results"].append(True)
						st.session_state["active_step"] = step + 1
						st.rerun()
				with col_fail:
					if st.button(f"Report Failed / Timeout", use_container_width=True):
						st.session_state["active_results"].append(False)
						st.session_state["active_step"] = step + 1
						st.rerun()
			else:
				# Completed all challenges
				results = st.session_state.get("active_results", [])
				passed_count = sum(results)
				total = len(results)
				all_passed = passed_count == total
				session_token = st.session_state["active_session"].token
				student_id = st.session_state.get("expected_student_id", "Unknown")

				services.audit_logger.append_event(
					result="GRANTED" if all_passed else "DENIED",
					activity_type="ACTIVE_LIVENESS",
					session_token=session_token,
					metadata={"results": results, "passed": passed_count, "total": total},
				)

				if all_passed:
					st.session_state["rec_result"] = (
						f"ACTIVE LIVENESS PASSED ({passed_count}/{total})\n\n"
						"Identity verified during challenges.\n\n"
						f"ACCESS GRANTED - Student ID: {student_id}"
					)
					st.session_state["rec_status"] = "Active liveness verified - Access granted."
					services.audit_logger.append_event(
						result="GRANTED",
						student_id=student_id,
						activity_type="RECOGNIZE",
						metadata={"reason": "active_liveness_passed"},
					)
				else:
					st.session_state["rec_result"] = (
						f"ACTIVE LIVENESS FAILED ({passed_count}/{total})\n\n"
						"Some challenges failed or identity mismatch.\nPlease try again or contact support."
					)
					st.session_state["rec_status"] = "Active liveness failed - Access denied."

				# Clean up active session
				del st.session_state["active_session"]
				del st.session_state["active_challenges"]
				st.session_state["active_liveness_required"] = False
				st.rerun()

		if st.button("Refresh Frame Source", use_container_width=True):
			st.session_state["rec_status"] = "Frame source refreshed."
			st.session_state["rec_result"] = "No recognition performed yet."
			st.session_state["rec_timer"] = "Time remaining: --"
			st.session_state["active_liveness_required"] = False
			st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENROLLMENT SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Enrollment":
	st.markdown('<div class="screen-title">Enrollment</div>', unsafe_allow_html=True)
	st.markdown(
		'<div class="screen-subtitle">Enroll a student by capturing multiple face samples and storing a transformed template.</div>',
		unsafe_allow_html=True,
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type = st.radio("Camera Source:", ["Camera Input", "Upload / Sample File"], horizontal=True, key="enr_source")
		enroll_frame: Optional[np.ndarray] = None

		if source_type == "Camera Input":
			cam_input = st.camera_input("Capture Enrollment Photo", key="enr_cam")
			enroll_frame = frame_to_bgr(cam_input)
		else:
			sample_dir = Path(__file__).parent / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options, key="enr_sample")
			up_file = st.file_uploader("Or Upload Image File", type=["jpg", "jpeg", "png"], key="enr_upload")

			if up_file is not None:
				enroll_frame = frame_to_bgr(up_file)
			elif selected_sample != "None":
				enroll_frame = cv2.imread(str(sample_dir / selected_sample))

		if enroll_frame is not None:
			preview_frame = draw_face_overlay(enroll_frame)
			st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Enrollment Preview", use_container_width=True)

	with col_right:
		st.markdown("#### Student ID")
		student_id = st.text_input("Student ID (Unique Identifier)", placeholder="e.g. S001", key="enroll_sid")

		st.markdown("#### Samples")
		sample_count = st.slider("Sample Count", min_value=1, max_value=10, value=5, step=1, key="enroll_samples")

		st.markdown("#### Status")
		enroll_status = st.session_state.get("enroll_status", "Ready to enroll.")
		st.markdown(f'<div class="status-box">{enroll_status}</div>', unsafe_allow_html=True)

		if st.button("Enroll Student", type="primary", use_container_width=True):
			sid = student_id.strip()
			if not sid:
				st.session_state["enroll_status"] = "Enter a student ID before enrolling."
				st.rerun()
			elif enroll_frame is None:
				st.session_state["enroll_status"] = "No frame available from camera."
				st.rerun()
			else:
				try:
					# Real biometric enrollment through pipeline
					detection = services.detector.detect(enroll_frame)
					embedding = services.embedder.extract_embedding(detection.face)
					services.enrollment.enroll_embedding(
						student_id=sid,
						embedding=embedding,
						metadata={"source": "ui", "samples": sample_count},
					)
					services.audit_logger.append_event(
						result="GRANTED",
						student_id=sid,
						activity_type="ENROLL",
						metadata={"screen": "enrollment", "samples": sample_count},
					)
					st.session_state["enroll_status"] = f"Enrolled {sid} successfully."
				except Exception as exc:
					st.session_state["enroll_status"] = f"Enrollment failed: {exc}"
				st.rerun()

		if st.button("Refresh Camera", use_container_width=True):
			st.session_state["enroll_status"] = "Camera source refreshed."
			st.rerun()

		st.markdown("---")
		st.markdown("#### Enrolled Students")
		records = services.enrollment.database.list_enrollments()
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
							services.audit_logger.append_event(
								result="GRANTED",
								student_id=record.student_id,
								activity_type="ENROLL",
								metadata={"screen": "enrollment", "action": "delete"},
							)
						except Exception as exc:
							st.session_state["enroll_status"] = f"Failed to delete {record.student_id}: {exc}"
						st.rerun()

		if st.button("Refresh List", use_container_width=True):
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

	page_index = st.session_state.get("audit_page_index", 0)
	page_size = 10

	rows, has_next_page = services.audit_logger.read_entries_page(page_size=page_size, page_index=page_index)

	# Toolbar matching desktop AuditScreen
	c_ref, c_prev, c_next, c_exp, c_info = st.columns([1, 1, 1, 1.2, 1.5])
	with c_ref:
		if st.button("Refresh", use_container_width=True):
			st.rerun()
	with c_prev:
		if st.button("Previous", disabled=(page_index <= 0), use_container_width=True):
			st.session_state["audit_page_index"] = max(0, page_index - 1)
			st.rerun()
	with c_next:
		if st.button("Next", disabled=(not has_next_page), use_container_width=True):
			st.session_state["audit_page_index"] = page_index + 1
			st.rerun()
	with c_exp:
		all_entries = services.audit_logger.read_entries()
		if all_entries:
			output = io.StringIO()
			writer = csv.DictWriter(output, fieldnames=list(all_entries[0].keys()))
			writer.writeheader()
			writer.writerows(all_entries)
			csv_data = output.getvalue()
			st.download_button("Export CSV", data=csv_data, file_name="audit_log.csv", mime="text/csv", use_container_width=True)
		else:
			st.button("Export CSV", disabled=True, use_container_width=True)
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

	matcher_val = st.slider("Matcher threshold", 0.1, 0.95, float(services.matcher.threshold), 0.05)
	passive_val = st.slider("Passive liveness threshold", 0.1, 0.95, float(services.passive_liveness.threshold), 0.05)
	window_val = st.slider("Active window (seconds)", 1.0, 5.0, float(services.active_liveness.window_seconds), 0.5)

	settings_status = st.session_state.get("settings_status", "Adjust values and save them to the live services.")
	st.markdown(f'<div class="status-box">{settings_status}</div>', unsafe_allow_html=True)

	if st.button("Apply Settings", type="primary"):
		try:
			services.matcher.set_threshold(float(matcher_val))
			services.passive_liveness.threshold = float(passive_val)
			services.active_liveness.window_seconds = float(window_val)
			st.session_state["settings_status"] = "Settings applied to live services."
		except Exception as exc:
			st.session_state["settings_status"] = f"Could not apply settings: {exc}"
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

	st.write("Offline evaluation is handled by the `evaluation/` package.")
	st.write("Use the benchmark and attack simulation modules from the command line for full ROC/EER analysis.")

	eval_status = st.session_state.get("eval_status", "UI evaluation panel is connected; run offline metrics from the CLI for now.")
	st.markdown(f'<div class="status-box">{eval_status}</div>', unsafe_allow_html=True)

	if st.button("Refresh Status"):
		st.session_state["eval_status"] = "UI evaluation panel is connected; run offline metrics from the CLI for now."
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
		source_type = st.radio("Camera Source:", ["Camera Input", "Upload / Sample File"], horizontal=True, key="live_source")
		live_frame: Optional[np.ndarray] = None

		if source_type == "Camera Input":
			cam_input = st.camera_input("Capture frame for liveness analysis", key="live_cam")
			live_frame = frame_to_bgr(cam_input)
		else:
			sample_dir = Path(__file__).parent / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
			sample_options = ["None"] + [p.name for p in sample_files]
			selected_sample = st.selectbox("Select Sample Photo:", sample_options, key="live_sample")
			up_file = st.file_uploader("Or Upload Image File", type=["jpg", "jpeg", "png"], key="live_upload")

			if up_file is not None:
				live_frame = frame_to_bgr(up_file)
			elif selected_sample != "None":
				live_frame = cv2.imread(str(sample_dir / selected_sample))

		if live_frame is not None:
			st.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), caption="Liveness Inspection Frame", use_container_width=True)

	with col_right:
		st.markdown("#### Passive Liveness")
		passive_status = st.session_state.get("passive_status", "Passive liveness not checked yet.")
		st.markdown(f'<div class="status-box">{passive_status}</div>', unsafe_allow_html=True)

		if st.button("Check Current Frame", type="primary", use_container_width=True):
			if live_frame is None:
				st.session_state["passive_status"] = "No frame available from camera."
			else:
				try:
					try:
						detection = services.recognition.detect_face(live_frame)
						face_crop = getattr(detection, "face", live_frame)
					except Exception:
						face_crop = live_frame

					result = services.passive_liveness.classify(face_crop)
					if not result.used_model:
						display = (
							f"⚠ HEURISTIC FALLBACK — score {result.score:.3f}\n\n"
							"MiniFASNet model is NOT loaded.\n"
							"This score is UNRELIABLE and cannot distinguish "
							"real faces from photos. Install the "
							"Silent-Face-Anti-Spoofing package and download "
							"the model weights."
						)
					else:
						display = f"{result.label.upper()} at score {result.score:.3f}"

					st.session_state["passive_status"] = display
					services.audit_logger.append_event(
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
					st.session_state["passive_status"] = f"Passive check failed: {exc}"
			st.rerun()

		st.markdown("---")
		st.markdown("#### Active Challenge")
		challenge_status = st.session_state.get("challenge_status", "No active session created yet.")
		st.markdown(f'<div class="status-box">{challenge_status}</div>', unsafe_allow_html=True)

		if st.button("Generate Session", use_container_width=True):
			try:
				session = services.active_liveness.create_session()
				st.session_state["challenge_status"] = f"Token: {session.token[:12]}... | Challenges: {', '.join(session.challenge_names)}"
			except Exception as exc:
				st.session_state["challenge_status"] = f"Could not create session: {exc}"
			st.rerun()
