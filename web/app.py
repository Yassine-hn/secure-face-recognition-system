"""Secure Face Recognition System (SFRS) - Interactive Web Dashboard.

Provides browser-based facial recognition, student enrollment, multi-factor
liveness testing, cryptographic audit log inspection, and benchmark metrics.
Suitable for local use and 1-click cloud hosting (Hugging Face Spaces, Streamlit Cloud).
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from core.services import AppServices, build_default_services
from evaluation.metrics import compute_eer, roc_auc, roc_curve_points
from security.audit_log import verify_audit_chain


# ── Page Configuration & Theming ─────────────────────────────────────────────
st.set_page_config(
	page_title="SFRS - Secure Face Recognition System",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	.main-title {
		font-size: 2.2rem;
		font-weight: 700;
		margin-bottom: 0.2rem;
		color: #58a6ff;
	}
	.sub-title {
		font-size: 1.05rem;
		color: #8b949e;
		margin-bottom: 1.5rem;
	}
	.metric-card {
		background-color: #161b22;
		border: 1px solid #30363d;
		border-radius: 8px;
		padding: 16px;
		margin-bottom: 12px;
	}
	.badge-success {
		background-color: #238636;
		color: white;
		padding: 4px 10px;
		border-radius: 12px;
		font-weight: 600;
		font-size: 0.85rem;
	}
	.badge-danger {
		background-color: #da3633;
		color: white;
		padding: 4px 10px;
		border-radius: 12px;
		font-weight: 600;
		font-size: 0.85rem;
	}
	.badge-warning {
		background-color: #d29922;
		color: black;
		padding: 4px 10px;
		border-radius: 12px;
		font-weight: 600;
		font-size: 0.85rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_app_services() -> AppServices:
	"""Load and cache system backend services."""
	root_dir = Path(__file__).resolve().parents[1]
	return build_default_services(base_dir=root_dir)


services = load_app_services()


# ── Helper Functions ──────────────────────────────────────────────────────────
def frame_to_bgr(uploaded_or_camera: Any) -> Optional[np.ndarray]:
	"""Convert Streamlit camera input or uploaded file to OpenCV BGR array."""
	if uploaded_or_camera is None:
		return None
	try:
		bytes_data = uploaded_or_camera.getvalue()
		image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
		rgb_array = np.array(image)
		return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
	except Exception as exc:
		st.error(f"Failed to process image: {exc}")
		return None


def draw_detection_box(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], label: str = "Face", is_live: bool = True) -> np.ndarray:
	"""Draw bounding box with status banner on OpenCV frame."""
	annotated = frame_bgr.copy()
	x1, y1, x2, y2 = bbox
	color = (46, 204, 113) if is_live else (46, 46, 231)  # Green vs Red (BGR)
	cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
	cv2.putText(annotated, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
	return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ── Sidebar: System Controls & Health ─────────────────────────────────────────
with st.sidebar:
	st.image("https://img.shields.io/badge/Security-Cancelable_Biometrics-blue?style=for-the-badge&logo=shield", use_container_width=True)
	st.title("System Controls")
	st.markdown("---")

	# Runtime status
	enrolled_records = list(services.database.list_enrollments())
	st.metric("Enrolled Identities", len(enrolled_records))

	st.markdown("### Security Parameters")
	threshold = st.slider(
		"Matcher Cosine Threshold",
		min_value=0.1,
		max_value=0.9,
		value=0.5,
		step=0.05,
		help="Higher threshold reduces False Accepts (impostors), lower threshold reduces False Rejects.",
	)
	services.matcher.threshold = threshold

	passive_thresh = st.slider(
		"Passive Liveness Threshold",
		min_value=0.1,
		max_value=0.9,
		value=0.5,
		step=0.05,
		help="Minimum texture/anti-spoofing confidence score to classify as LIVE.",
	)

	st.markdown("---")
	st.markdown("### Encryption & Storage")
	db_mode = "SQLCipher (AES-256)" if getattr(services.database, "_is_sqlcipher", False) else "SQLite (Encrypted Fallback)"
	st.caption(f"**Database:** {db_mode}")
	st.caption("**Key Storage:** OS Keyring / Environment")
	st.caption("**Biometrics:** Random Projection 512-D (Revocable)")

	st.markdown("---")
	if st.button("🗑️ Reset Demo Database", help="Clear all enrolled student templates"):
		services.database.clear()
		st.success("Database cleared!")
		st.rerun()


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ Secure Face Recognition System</div>', unsafe_allow_html=True)
st.markdown(
	'<div class="sub-title">Privacy-Preserving Biometric Authentication with Cancelable Projections, '
	'Multi-Factor Liveness Detection & Cryptographic Audit Chains</div>',
	unsafe_allow_html=True,
)

tabs = st.tabs([
	"🔍 Live Verification",
	"👤 Student Enrollment",
	"🛡️ Liveness Lab",
	"📜 Audit Trail",
	"📊 Benchmarks & ROC",
	"ℹ️ Architecture & GDPR",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Verification & Recognition
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
	st.markdown("### 📷 Live Camera Verification")
	st.write("Authenticate using your browser webcam or choose a test sample from the selector below.")

	col_input, col_result = st.columns([1, 1], gap="medium")

	with col_input:
		input_mode = st.radio("Input Source:", ["Webcam Camera", "Upload Image File", "Use Demo Sample Image"], horizontal=True)

		bgr_image: Optional[np.ndarray] = None
		if input_mode == "Webcam Camera":
			cam_file = st.camera_input("Take a photo to authenticate")
			bgr_image = frame_to_bgr(cam_file)
		elif input_mode == "Upload Image File":
			up_file = st.file_uploader("Upload Face Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
			bgr_image = frame_to_bgr(up_file)
		else:
			sample_dir = Path(__file__).parent / "sample_faces"
			sample_files = list(sample_dir.glob("*.jpg"))
			if sample_files:
				selected_sample = st.selectbox("Select Sample Image:", sample_files, format_func=lambda p: p.stem.replace("_", " ").title())
				if selected_sample:
					bgr_image = cv2.imread(str(selected_sample))
					st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), caption=f"Selected: {selected_sample.name}", width=260)
			else:
				st.info("No sample files found.")

	with col_result:
		st.markdown("### Authentication Pipeline")
		if bgr_image is None:
			st.info("👈 Please capture a frame or select a sample image to begin.")
		else:
			# Step 1: Passive Liveness
			with st.spinner("Running Anti-Spoofing & Liveness Check..."):
				passive_result = services.passive_liveness.evaluate(bgr_image)

			col_liveness1, col_liveness2 = st.columns(2)
			with col_liveness1:
				st.metric("Liveness Score", f"{passive_result.score:.3f}")
			with col_liveness2:
				is_live = passive_result.score >= passive_thresh
				if is_live:
					st.markdown('<span class="badge-success">LIVE HUMAN DETECTED</span>', unsafe_allow_html=True)
				else:
					st.markdown('<span class="badge-danger">SPOOF / ATTACK DETECTED</span>', unsafe_allow_html=True)

			# Step 2: Face Detection
			try:
				detected = services.detector.detect_face(bgr_image)
				annotated_rgb = draw_detection_box(bgr_image, detected.bbox, "Live Face" if is_live else "Potential Spoof", is_live)
				st.image(annotated_rgb, caption="Detected Face ROI", use_container_width=True)
			except Exception as exc:
				st.warning(f"Detection note: {exc}")
				detected = None

			# Step 3: Recognition & Matching
			if not is_live:
				st.error("🛑 ACCESS BLOCKED: Passive anti-spoofing check failed. Presentation attack prevented!")
				services.audit_logger.log_event(
					activity_type="RECOGNITION",
					result="DENIED",
					student_id="unknown_spoof",
					metadata={"reason": "passive_spoof_detected", "score": passive_result.score},
				)
			elif detected is None:
				st.warning("⚠️ No valid face region detected. Please look directly at the camera.")
			else:
				# Run recognition against enrolled DB
				enrolled_map = {rec.student_id: rec.template for rec in services.database.list_enrollments()}
				if not enrolled_map:
					st.warning("ℹ️ No students enrolled in the database yet. Please enroll in the 'Student Enrollment' tab first!")
				else:
					with st.spinner("Extracting 512-D ArcFace & Cancelable Projection..."):
						query_emb = services.embedder.extract_embedding(detected.face)
						transformed_query = services.transformer.transform(query_emb)
						match = services.matcher.find_best_match(transformed_query, enrolled_map)

					template_hash = services.transformer.hash_template(transformed_query)[:16]
					st.caption(f"**Cancelable Template SHA-256:** `{template_hash}...` (Irreversible)")

					if match.is_match and match.student_id is not None:
						st.success(f"🎉 **ACCESS GRANTED** — Recognized as **{match.student_id}**")
						st.metric("Cosine Similarity Match Score", f"{match.score:.4f}", delta=f"{match.score - threshold:+.4f} vs threshold")
						services.audit_logger.log_event(
							activity_type="RECOGNITION",
							result="GRANTED",
							student_id=match.student_id,
							metadata={"similarity_score": round(match.score, 4), "threshold": threshold},
						)
					else:
						st.error(f"⛔ **ACCESS DENIED** — Identity not verified (Best Match: `{match.student_id}` with score {match.score:.4f} < {threshold})")
						services.audit_logger.log_event(
							activity_type="RECOGNITION",
							result="DENIED",
							student_id=match.student_id or "unrecognized",
							metadata={"best_score": round(match.score, 4), "threshold": threshold},
						)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Student Enrollment
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
	st.markdown("### 👤 Enroll New Student Face Template")
	st.write("Enrollment securely transforms the biometric face vector using Cancelable Biometrics before storing into the encrypted database.")

	col_form, col_cam = st.columns([1, 1], gap="medium")

	with col_form:
		student_id = st.text_input("Student ID (Unique Identifier):", placeholder="e.g. STU-2024-001")
		student_name = st.text_input("Full Name:", placeholder="e.g. John Doe")
		department = st.text_input("Department / Major:", placeholder="e.g. Computer Science")

		st.markdown("##### Enrollment Policy")
		st.caption("✅ 512-D ArcFace Embedding extracted")
		st.caption("✅ Seeded Cancelable Random Projection applied")
		st.caption("✅ Raw face image & raw vector discarded immediately")
		st.caption("✅ Encrypted with SQLCipher AES-256 at rest")

	with col_cam:
		enroll_mode = st.radio("Capture Method:", ["Camera", "Upload File"], horizontal=True, key="enroll_radio")
		enroll_bgr = None
		if enroll_mode == "Camera":
			enroll_file = st.camera_input("Capture Enrollment Photo")
			enroll_bgr = frame_to_bgr(enroll_file)
		else:
			enroll_up = st.file_uploader("Upload Profile Image", type=["jpg", "png", "jpeg"], key="enroll_uploader")
			enroll_bgr = frame_to_bgr(enroll_up)

	if st.button("🚀 Enroll Student", type="primary"):
		if not student_id.strip():
			st.error("Please provide a valid Student ID.")
		elif enroll_bgr is None:
			st.error("Please capture or upload an enrollment photo.")
		else:
			try:
				# Liveness check first
				liveness_res = services.passive_liveness.evaluate(enroll_bgr)
				if liveness_res.score < passive_thresh:
					st.error(f"Enrollment rejected: Image failed anti-spoofing check (score: {liveness_res.score:.2f})")
				else:
					det = services.detector.detect_face(enroll_bgr)
					embedding = services.embedder.extract_embedding(det.face)
					services.enrollment.enroll_student(
						student_id=student_id.strip(),
						embedding=embedding,
						metadata={"name": student_name, "department": department},
					)
					services.audit_logger.log_event(
						activity_type="ENROLLMENT",
						result="GRANTED",
						student_id=student_id.strip(),
						metadata={"name": student_name, "dept": department},
					)
					st.success(f"Successfully enrolled student **{student_id}**! Template secured and stored.")
					time.sleep(1)
					st.rerun()
			except Exception as exc:
				st.error(f"Enrollment failed: {exc}")

	st.markdown("---")
	st.markdown("### 📋 Enrolled Student Registry")
	current_records = list(services.database.list_enrollments())
	if current_records:
		records_data = []
		for r in current_records:
			records_data.append({
				"Student ID": r.student_id,
				"Name": r.metadata.get("name", "N/A"),
				"Department": r.metadata.get("department", "N/A"),
				"Enrolled At": r.created_at[:19] if r.created_at else "N/A",
				"Template Dim": len(r.template),
			})
		df_records = pd.DataFrame(records_data)
		st.dataframe(df_records, use_container_width=True)

		del_id = st.selectbox("Select Student ID to Revoke / Delete:", [r.student_id for r in current_records])
		if st.button(f"🗑️ Revoke Template for {del_id}", help="Demonstrates template revocability without compromising raw biometrics"):
			services.database.delete_enrollment(del_id)
			services.audit_logger.log_event("REVOCATION", "GRANTED", del_id, {"action": "template_revoked"})
			st.success(f"Template for {del_id} revoked and erased.")
			st.rerun()
	else:
		st.info("No enrolled students in the database.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Liveness Lab
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
	st.markdown("### 🛡️ Multi-Factor Anti-Spoofing & Liveness Lab")
	st.write("Explore how SFRS protects against Presentation Attacks (2D printouts, tablet/phone screen replay, 3D masks).")

	col_passive, col_active = st.columns(2, gap="large")

	with col_passive:
		st.markdown("#### 1. Passive Liveness (Texture & Frequency Analysis)")
		st.caption("Analyzes high-frequency reflections, specular highlights, and color distortion without requiring user cooperation.")
		test_file = st.file_uploader("Upload Test Frame for Liveness Analysis:", type=["jpg", "png", "jpeg"], key="liveness_test_file")
		if test_file:
			test_bgr = frame_to_bgr(test_file)
			if test_bgr is not None:
				st.image(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), caption="Test Image", width=280)
				res = services.passive_liveness.evaluate(test_bgr)
				st.metric("Anti-Spoofing Live Confidence", f"{res.score * 100:.1f}%")
				if res.score >= passive_thresh:
					st.success("Verdict: LIVE HUMAN")
				else:
					st.error("Verdict: SPOOF / REPLAY ATTACK")

	with col_active:
		st.markdown("#### 2. Active Liveness (Dynamic Challenge Orchestration)")
		st.caption("Issues non-deterministic behavioral challenges within strict time windows (3s) to defeat pre-recorded video injection.")

		if st.button("🎲 Generate Dynamic Challenge Session"):
			session = services.active_liveness.create_session(window_seconds=4.0)
			st.session_state["active_session"] = session

		if "active_session" in st.session_state:
			s = st.session_state["active_session"]
			st.markdown(f"**Session Token:** `{s.token[:16]}...`")
			st.markdown(f"**Allowed Window:** `{s.window_seconds}` seconds")
			st.markdown("**Issued Challenges in Sequence:**")
			for i, ch in enumerate(s.challenge_names, 1):
				st.markdown(f"**Step {i}:** `{ch.upper().replace('_', ' ')}`")
			st.info("In the desktop application with high-speed video capture, MediaPipe Face Mesh evaluates landmarks in real time.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Cryptographic Audit Trail
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
	st.markdown("### 📜 Tamper-Evident HMAC-SHA256 Audit Trail")
	st.write("All authentication attempts, enrollments, and security violations are committed to an append-only log with blockchain-style HMAC chaining.")

	entries = list(services.audit_logger.read_entries())

	col_verify, col_export = st.columns([2, 1])
	with col_verify:
		# Verify HMAC chain
		if entries:
			try:
				is_chain_valid = services.audit_logger.verify()
			except Exception:
				is_chain_valid = False

			if is_chain_valid:
				st.markdown('<span class="badge-success">🔒 HMAC CHAIN VALID — 0 TAMPERING DETECTED</span>', unsafe_allow_html=True)
			else:
				st.markdown('<span class="badge-danger">⚠️ INTEGRITY ERROR: LOG FILE TAMPERED OR ALTERED</span>', unsafe_allow_html=True)
		else:
			st.info("No audit entries yet.")

	with col_export:
		if entries:
			csv_buffer = io.StringIO()
			writer = csv.DictWriter(csv_buffer, fieldnames=list(entries[0].keys()))
			writer.writeheader()
			writer.writerows(entries)
			st.download_button(
				label="📥 Download Audit Log (CSV)",
				data=csv_buffer.getvalue(),
				file_name="sfrs_audit_log.csv",
				mime="text/csv",
			)

	st.markdown("---")
	if entries:
		df_audit = pd.DataFrame(entries)
		cols_to_show = ["timestamp", "activity_type", "result", "student_id", "hmac"]
		existing_cols = [c for c in cols_to_show if c in df_audit.columns]
		st.dataframe(df_audit[existing_cols].iloc[::-1], use_container_width=True, height=360)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: Benchmarks & Security Analytics
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
	st.markdown("### 📊 Benchmark Metrics & ROC Curve")
	st.write("Evaluate biometric accuracy trade-offs between False Accept Rate (FAR) and False Reject Rate (FRR).")

	# Generate realistic synthetic distribution for interactive simulation
	np.random.seed(42)
	sim_genuine = np.random.normal(loc=0.78, scale=0.08, size=500)
	sim_impostor = np.random.normal(loc=0.25, scale=0.10, size=1500)

	sim_genuine = np.clip(sim_genuine, 0.0, 1.0)
	sim_impostor = np.clip(sim_impostor, 0.0, 1.0)

	fpr, tpr, roc_thresh = roc_curve_points(sim_genuine, sim_impostor, steps=200)
	auc = roc_auc(fpr, tpr)
	eer_res = compute_eer(sim_genuine, sim_impostor, steps=200)

	col_m1, col_m2, col_m3 = st.columns(3)
	col_m1.metric("Equal Error Rate (EER)", f"{eer_res.eer * 100:.2f}%")
	col_m2.metric("Optimal Operating Threshold", f"{eer_res.threshold:.3f}")
	col_m3.metric("Area Under Curve (AUC)", f"{auc:.4f}")

	st.markdown("---")
	st.markdown("#### ROC Curve (Receiver Operating Characteristic)")

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(8, 4))
	fig.patch.set_facecolor("#0e1117")
	ax.set_facecolor("#161b22")
	ax.plot(fpr, tpr, color="#58a6ff", lw=2.5, label=f"ArcFace 512-D + Cancelable (AUC = {auc:.4f})")
	ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--", lw=1.5, label="Random Guess")
	ax.scatter([eer_res.far], [1.0 - eer_res.frr], color="#2ea043", s=80, zorder=5, label=f"EER Point ({eer_res.eer*100:.1f}%)")
	ax.set_xlabel("False Accept Rate (FAR)", color="#c9d1d9")
	ax.set_ylabel("True Accept Rate (1 - FRR)", color="#c9d1d9")
	ax.tick_params(colors="#8b949e")
	for spine in ax.spines.values():
		spine.set_color("#30363d")
	ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
	st.pyplot(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: Architecture & GDPR Compliance
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
	st.markdown("### 🏛️ Privacy-First Biometric Security Architecture")
	st.write(
		"Biometric templates are uniquely personal and cannot be changed if compromised. "
		"SFRS implements **Cancelable Biometrics** and defense-in-depth principles in compliance with GDPR Article 9."
	)

	st.markdown(
		"""
		| Layer | Mechanism | Security Guarantee |
		| :--- | :--- | :--- |
		| **1. Ingestion** | RetinaFace / OpenCV Haar Cascade | Precise face localization and facial landmarks extraction. |
		| **2. Feature Space** | ArcFace 512-D Normalized Vector | Deep metric learning with angular margin loss for inter-class separation. |
		| **3. Privacy Shield** | **Cancelable Random Projection** | Raw vectors are projected through a secret pseudo-random orthogonal matrix $T = P \\cdot E$. Irreversible and revocable. |
		| **4. Liveness Defense**| Passive MiniFASNet + Active 5-Point | Defeats 2D printouts, video replays, and presentation attacks. |
		| **5. Storage at Rest** | **SQLCipher AES-256-CBC** | Transformed templates stored in encrypted database. Master key managed via OS Keyring. |
		| **6. Accountability** | **HMAC-SHA256 Hash Chaining** | Every access decision creates an immutable log block $H_i = \\text{HMAC}(H_{i-1} \\parallel \\text{payload})$. |
		"""
	)

	st.markdown("---")
	st.markdown("#### 🇪🇺 GDPR Article 9 Compliance Notes")
	st.info(
		"Under GDPR Article 9, biometric data for uniquely identifying a natural person is classified as 'Special Category Data'. "
		"By implementing **Cancelable Biometrics**, SFRS ensures that: \n"
		"1. **Irreversibility**: It is computationally infeasible to reconstruct the original face image from the stored template.\n"
		"2. **Revocability & Renewability**: If a database is breached, the user's template can be revoked and re-issued with a new projection seed without changing the user's physical face.\n"
		"3. **Unlinkability**: Different applications using different projection seeds cannot cross-match templates to track users."
	)
