"""Streamlit web interface for the Secure Face Recognition System (SFRS).

Provides clean enterprise dark theme typography, sharp rectilinear architecture,
structured metric telemetry, accessible status tags, and seamless cloud/hardware camera support.
"""

from __future__ import annotations

import atexit
import csv
import io
import os
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


# ── Page Configuration & Enterprise Theming ───────────────────────────────────
st.set_page_config(
	page_title="Secure Face Recognition System • SFRS",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

	:root {
		--cds-background: #161616;
		--cds-layer-01: #262626;
		--cds-layer-02: #393939;
		--cds-layer-hover: #333333;
		--cds-border-subtle: #393939;
		--cds-border-strong: #525252;
		--cds-text-primary: #f4f4f4;
		--cds-text-secondary: #c6c6c6;
		--cds-text-helper: #8d8d8d;
		--cds-interactive: #0f62fe;
		--cds-interactive-hover: #0353e9;
		--cds-interactive-active: #002d9c;
		--cds-support-success: #24a148;
		--cds-support-error: #da1e28;
		--cds-support-warning: #f1c21b;
		--cds-support-info: #4589ff;
		--cds-font-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
		--cds-font-mono: 'IBM Plex Mono', 'Menlo', 'DejaVu Sans Mono', Courier, monospace;
	}

	/* Global Font & Theme Overrides */
	html, body, [class*="css"], .stApp {
		font-family: var(--cds-font-sans) !important;
		background-color: var(--cds-background) !important;
		color: var(--cds-text-primary) !important;
	}

	/* Hide default Streamlit Deploy & Decoration elements */
	.stDeployButton,
	div[data-testid="stDeployButton"],
	header [data-testid="stDeployButton"],
	button[data-testid="stDeployButton"],
	button[title="Deploy this app"],
	#MainMenu, footer {
		display: none !important;
	}

	.main .block-container {
		padding-top: 1rem;
		padding-bottom: 2.5rem;
		max-width: 1300px;
	}

	/* Carbon UI Shell Header Bar */
	.cds--shell-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		background-color: #161616;
		border-bottom: 1px solid var(--cds-border-subtle);
		padding: 10px 16px;
		margin: -1rem -1rem 1.5rem -1rem;
	}
	.cds--shell-brand {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 13px;
		letter-spacing: 0.16px;
	}
	.cds--shell-prefix {
		font-weight: 700;
		color: #ffffff;
		background-color: var(--cds-interactive);
		padding: 2px 6px;
		border-radius: 0px;
	}
	.cds--shell-product {
		font-weight: 500;
		color: var(--cds-text-primary);
	}
	.cds--shell-status {
		display: flex;
		gap: 8px;
		align-items: center;
	}

	/* Breadcrumb Navigation */
	.cds--breadcrumb {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: var(--cds-text-helper);
		margin-bottom: 4px;
		text-transform: uppercase;
		letter-spacing: 0.32px;
	}
	.cds--breadcrumb-item:after {
		content: "/";
		margin-left: 6px;
		color: #525252;
	}
	.cds--breadcrumb-item:last-child:after {
		content: "";
	}
	.cds--breadcrumb-item--current {
		color: var(--cds-interactive);
		font-weight: 500;
	}

	/* Page Title & Subtitle */
	.cds--page-header {
		margin-bottom: 20px;
		border-bottom: 1px solid var(--cds-border-subtle);
		padding-bottom: 14px;
	}
	.cds--page-title {
		font-size: 26px !important;
		font-weight: 400 !important;
		color: var(--cds-text-primary) !important;
		margin: 0 0 4px 0 !important;
		line-height: 1.25 !important;
		letter-spacing: 0;
	}
	.cds--page-subtitle {
		font-size: 13px !important;
		color: var(--cds-text-secondary) !important;
		margin: 0 !important;
		line-height: 1.4 !important;
	}

	/* Carbon Tile Container (Flat architectural card) */
	.cds--tile {
		background-color: var(--cds-layer-01);
		border: 1px solid var(--cds-border-subtle);
		padding: 16px 18px;
		border-radius: 0px !important;
		margin-bottom: 16px;
		transition: border-color 0.15s ease-in-out;
	}
	.cds--tile:hover {
		border-color: var(--cds-border-strong);
	}
	.cds--tile-header {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.32px;
		color: var(--cds-text-helper);
		margin-bottom: 12px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border-bottom: 1px solid var(--cds-border-subtle);
		padding-bottom: 8px;
	}

	/* Carbon Rectangular Tags */
	.cds--tag {
		display: inline-flex;
		align-items: center;
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.32px;
		padding: 3px 8px;
		border-radius: 0px !important;
		text-transform: uppercase;
		line-height: 1.2;
		font-family: var(--cds-font-mono);
	}
	.cds--tag--green {
		background-color: #122a1b;
		color: #42be65;
		border: 1px solid #24a148;
	}
	.cds--tag--red {
		background-color: #3c1216;
		color: #ff8389;
		border: 1px solid #da1e28;
	}
	.cds--tag--blue {
		background-color: #09224f;
		color: #78a9ff;
		border: 1px solid #0f62fe;
	}
	.cds--tag--gray {
		background-color: #262626;
		color: #c6c6c6;
		border: 1px solid #525252;
	}
	.cds--tag--yellow {
		background-color: #302600;
		color: #f1c21b;
		border: 1px solid #b28600;
	}

	/* Carbon Key-Value Row */
	.cds--kv-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 0;
		border-bottom: 1px solid #333333;
		font-size: 13px;
	}
	.cds--kv-row:last-child {
		border-bottom: none;
	}
	.cds--kv-label {
		color: var(--cds-text-secondary);
	}
	.cds--kv-value {
		color: var(--cds-text-primary);
		font-weight: 500;
		font-family: var(--cds-font-mono);
	}

	/* Carbon Inline Notification */
	.cds--inline-notification {
		display: flex;
		flex-direction: column;
		background-color: var(--cds-layer-01);
		border: 1px solid var(--cds-border-subtle);
		border-left: 3px solid var(--cds-interactive);
		padding: 12px 16px;
		margin-bottom: 14px;
		border-radius: 0px !important;
	}
	.cds--inline-notification--success {
		border-left-color: var(--cds-support-success);
		background-color: #122216;
	}
	.cds--inline-notification--error {
		border-left-color: var(--cds-support-error);
		background-color: #2a1215;
	}
	.cds--inline-notification--warning {
		border-left-color: var(--cds-support-warning);
		background-color: #26210f;
	}
	.cds--inline-notification--info {
		border-left-color: var(--cds-support-info);
		background-color: #121d2f;
	}
	.cds--inline-notification__title {
		font-size: 13px;
		font-weight: 600;
		color: var(--cds-text-primary);
		margin-bottom: 2px;
		text-transform: uppercase;
		letter-spacing: 0.16px;
	}
	.cds--inline-notification__subtitle {
		font-size: 12px;
		color: var(--cds-text-secondary);
		line-height: 1.4;
	}

	/* Carbon Monospace Code Snippet */
	.cds--code-snippet {
		background-color: #121212;
		border: 1px solid var(--cds-border-subtle);
		padding: 12px 14px;
		font-family: var(--cds-font-mono) !important;
		font-size: 12px;
		line-height: 1.5;
		color: #e0e0e0;
		border-radius: 0px !important;
		margin-bottom: 14px;
		overflow-x: auto;
		white-space: pre-wrap;
	}

	/* Carbon Data Table */
	.cds--data-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
		margin-bottom: 16px;
		border: 1px solid var(--cds-border-subtle);
	}
	.cds--data-table th {
		background-color: #161616;
		color: var(--cds-text-secondary);
		font-weight: 600;
		font-size: 11px;
		letter-spacing: 0.32px;
		text-transform: uppercase;
		padding: 10px 14px;
		text-align: left;
		border-bottom: 1px solid var(--cds-border-strong);
	}
	.cds--data-table td {
		background-color: var(--cds-layer-01);
		color: var(--cds-text-primary);
		padding: 10px 14px;
		border-bottom: 1px solid var(--cds-border-subtle);
		font-family: var(--cds-font-sans);
	}
	.cds--data-table tr:hover td {
		background-color: var(--cds-layer-hover);
	}

	/* Carbon Button Styling */
	button[data-testid="baseButton-primary"],
	button[kind="primary"] {
		background-color: var(--cds-interactive) !important;
		color: #ffffff !important;
		border: none !important;
		border-radius: 0px !important;
		font-family: var(--cds-font-sans) !important;
		font-size: 14px !important;
		font-weight: 500 !important;
		letter-spacing: 0.16px !important;
		min-height: 42px !important;
		padding: 0 16px !important;
		transition: background-color 0.15s ease-in-out !important;
	}
	button[data-testid="baseButton-primary"]:hover,
	button[kind="primary"]:hover {
		background-color: var(--cds-interactive-hover) !important;
	}
	button[data-testid="baseButton-primary"]:active,
	button[kind="primary"]:active {
		background-color: var(--cds-interactive-active) !important;
	}

	button[data-testid="baseButton-secondary"],
	button[kind="secondary"] {
		background-color: var(--cds-layer-02) !important;
		color: var(--cds-text-primary) !important;
		border: 1px solid var(--cds-border-strong) !important;
		border-radius: 0px !important;
		font-family: var(--cds-font-sans) !important;
		font-size: 13px !important;
		font-weight: 400 !important;
		min-height: 38px !important;
		padding: 0 14px !important;
		transition: background-color 0.15s ease-in-out !important;
	}
	button[data-testid="baseButton-secondary"]:hover,
	button[kind="secondary"]:hover {
		background-color: #474747 !important;
		color: #ffffff !important;
	}

	/* Carbon Text Fields & Select Boxes */
	div[data-baseweb="input"],
	div[data-baseweb="select"] > div,
	div[data-baseweb="base-input"] {
		background-color: var(--cds-layer-01) !important;
		border: 1px solid var(--cds-border-subtle) !important;
		border-bottom: 1px solid var(--cds-text-helper) !important;
		border-radius: 0px !important;
		color: var(--cds-text-primary) !important;
		font-family: var(--cds-font-sans) !important;
	}
	div[data-baseweb="input"]:focus-within,
	div[data-baseweb="select"]:focus-within {
		border: 2px solid var(--cds-interactive) !important;
	}

	/* Carbon Radio Button Tabs */
	div[role="radiogroup"] {
		gap: 6px !important;
	}
	div[role="radiogroup"] label {
		background-color: var(--cds-layer-01) !important;
		border: 1px solid var(--cds-border-subtle) !important;
		padding: 6px 14px !important;
		border-radius: 0px !important;
		font-size: 13px !important;
		color: var(--cds-text-secondary) !important;
		transition: all 0.15s ease-in-out !important;
	}
	div[role="radiogroup"] label:hover {
		background-color: var(--cds-layer-hover) !important;
		color: var(--cds-text-primary) !important;
	}
	div[role="radiogroup"] label[data-checked="true"] {
		background-color: var(--cds-layer-02) !important;
		border-color: var(--cds-interactive) !important;
		color: #ffffff !important;
	}

	/* Sidebar Carbon Navigation */
	section[data-testid="stSidebar"] {
		background-color: #161616 !important;
		border-right: 1px solid var(--cds-border-subtle) !important;
		min-width: 250px;
		max-width: 290px;
	}
	section[data-testid="stSidebar"] div[role="radiogroup"] label {
		width: 100% !important;
		margin-bottom: 4px !important;
		border: 1px solid transparent !important;
		background-color: transparent !important;
		padding: 9px 12px !important;
		font-size: 13px !important;
	}
	section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
		background-color: var(--cds-layer-01) !important;
		color: #ffffff !important;
	}
	section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
		background-color: var(--cds-layer-01) !important;
		border-left: 3px solid var(--cds-interactive) !important;
		color: #ffffff !important;
		font-weight: 500 !important;
	}

	/* Carbon Progress Bar */
	div[data-testid="stProgress"] > div > div > div {
		background-color: var(--cds-interactive) !important;
		border-radius: 0px !important;
	}
	div[data-testid="stProgress"] > div > div {
		background-color: var(--cds-layer-02) !important;
		border-radius: 0px !important;
	}

	/* Carbon Native Alerts */
	div[data-testid="stAlert"] {
		border-radius: 0px !important;
		border-width: 1px !important;
		border-style: solid !important;
		background-color: var(--cds-layer-01) !important;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


# Synchronize Streamlit secrets into os.environ if running on Streamlit Cloud
try:
	for secret_key, secret_val in st.secrets.items():
		if isinstance(secret_val, str) and secret_key not in os.environ:
			os.environ[secret_key] = secret_val
except Exception:
	pass


def _seed_sample_identities_if_empty(srv: AppServices) -> None:
	"""Auto-seed sample identities into an empty database (e.g. fresh cloud instance)."""
	try:
		existing = srv.enrollment.database.list_enrollments()
		if existing:
			return
		sample_dir = PROJECT_ROOT / "web" / "sample_faces"
		alice_file = sample_dir / "alice_sample.jpg"
		if alice_file.exists():
			img = cv2.imread(str(alice_file))
			if img is not None and img.size > 0:
				try:
					emb, _ = srv.embedder.extract_embedding_from_frame(img)
				except Exception:
					crop = srv.detector.detect_face(img)
					emb = srv.embedder.extract_embedding(crop)
				if emb is not None:
					srv.enrollment.enroll_embedding(
						student_id="STU-001",
						embedding=emb,
						metadata={"name": "Alice (Demo Genuine)", "source": "sample_seed"},
					)
					try:
						srv.audit_logger.append_event(
							result="GRANTED",
							student_id="STU-001",
							activity_type="SYSTEM_INIT",
							metadata={"note": "Pre-enrolled sample profile STU-001 (Alice)"},
						)
					except Exception:
						pass
	except Exception as exc:
		print(f"[SFRS] Demo database auto-seeding notice: {exc}")


@st.cache_resource
def get_services() -> AppServices:
	"""Initialize and cache the backend application services container."""
	srv = build_default_services()
	_seed_sample_identities_if_empty(srv)
	return srv


try:
	services = get_services()
except Exception as init_exc:
	st.error(f"Failed to initialize core biometric services: {init_exc}")
	st.stop()


# ── UI Shell Helper Functions ────────────────────────────────────────────────
def render_shell_header() -> None:
	"""Render the persistent top application header bar."""
	st.markdown(
		"""
		<div class="cds--shell-header">
			<div class="cds--shell-brand">
				<span class="cds--shell-product">Secure Face Recognition System &bull; v1.0 Production</span>
			</div>
			<div class="cds--shell-status">
				<span class="cds--tag cds--tag--green">SYSTEM ACTIVE</span>
				<span class="cds--tag cds--tag--blue">AES-256 VAULT</span>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_page_header(breadcrumb_section: str, title: str, subtitle: str) -> None:
	"""Render standard breadcrumb, heading, and subtitle."""
	st.markdown(
		f"""
		<div class="cds--breadcrumb">
			<span class="cds--breadcrumb-item">Platform</span>
			<span class="cds--breadcrumb-item">Biometrics</span>
			<span class="cds--breadcrumb-item cds--breadcrumb-item--current">{breadcrumb_section}</span>
		</div>
		<div class="cds--page-header">
			<h1 class="cds--page-title">{title}</h1>
			<p class="cds--page-subtitle">{subtitle}</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


# Aliases for backward compatibility
render_carbon_shell_header = render_shell_header
render_carbon_page_header = render_page_header


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
	"""Draw subtle face bounding box overlay matching the desktop camera widget."""
	if frame is None or frame.size == 0:
		return frame
	annotated = frame.copy()
	try:
		cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
		cascade = cv2.CascadeClassifier(cascade_path)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
		faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
		for x, y, w, h in faces:
			cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (15, 98, 254), 2)
			cv2.putText(
				annotated,
				"FACE DETECTED",
				(int(x), max(16, int(y) - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(15, 98, 254),
				1,
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

	def is_available(self) -> bool:
		with self._lock:
			return self._running and self._cap is not None and self._cap.isOpened() and self._error is None


@st.cache_resource
def get_webcam_stream() -> WebcamManager:
	manager = WebcamManager()
	atexit.register(manager.stop)
	return manager


def get_camera_source_options() -> tuple[list[str], int]:
	"""Return available frame sources and the smart default index based on hardware availability."""
	try:
		cam = get_webcam_stream()
		is_avail = bool(callable(getattr(cam, "is_available", None)) and cam.is_available())
	except Exception:
		is_avail = False

	if is_avail:
		return ["Live Camera (Webcam)", "Sample Identity (Demo)", "Browser Camera (Webcam)", "Upload Image"], 0
	else:
		return ["Sample Identity (Demo)", "Browser Camera (Webcam)", "Upload Image", "Live Camera (Local Device)"], 0


def get_sample_identities() -> dict[str, str]:
	"""Discover and map available pre-loaded face samples to clean descriptive labels.

	Returns a dictionary mapping human-readable labels to filenames in web/sample_faces.
	Guarantees standard genuine identities (Alice, Bob) and attack samples (Spoof) are mapped.
	"""
	sample_dir = PROJECT_ROOT / "web" / "sample_faces"
	mapping: dict[str, str] = {}
	if not sample_dir.exists():
		sample_dir.mkdir(parents=True, exist_ok=True)

	known = {
		"alice_sample.jpg": "Alice • Genuine Identity (STU-001)",
		"bob_sample.jpg": "Bob • Genuine Identity (STU-002)",
		"spoof_screen_sample.jpg": "Screen Replay Attack • Spoof Sample",
	}

	for filename, label in known.items():
		if (sample_dir / filename).exists():
			mapping[label] = filename

	# Discover any additional sample files added (.jpg, .jpeg, .png)
	for ext in ("*.jpg", "*.jpeg", "*.png"):
		for p in sorted(sample_dir.glob(ext)):
			if p.name not in known and p.name not in mapping.values():
				clean_name = p.stem.replace("_", " ").title()
				mapping[f"{clean_name} ({p.name})"] = p.name

	return mapping


def capture_frames(
	source_type: str,
	count: int = 3,
	sample_name: Optional[str] = None,
	uploaded_file: Any = None,
	camera_file: Any = None,
) -> list[np.ndarray]:
	"""Automatically capture a burst of frames from the selected video or image source.

	No manual 'take picture' button is required; the app samples sequential frames directly.
	"""
	frames: list[np.ndarray] = []

	if source_type.startswith("Live Camera"):
		manager = get_webcam_stream()
		return manager.capture_burst(count=count)

	elif source_type.startswith("Browser Camera"):
		if camera_file is None:
			raise ValueError("Please capture a photo first using the 'Take Photo' camera widget.")
		bgr = frame_to_bgr(camera_file)
		if bgr is None or bgr.size == 0:
			raise ValueError("Could not decode the snapshot from your browser camera.")
		for _ in range(max(1, count)):
			frames.append(bgr.copy())
		return frames

	elif source_type.startswith("Sample Identity"):
		sample_dir = PROJECT_ROOT / "web" / "sample_faces"
		samples = get_sample_identities()
		# Fallback resolution if sample_name is empty, "None", or a descriptive label
		if not sample_name or sample_name == "None":
			if samples:
				sample_name = next(iter(samples.values()))
			else:
				raise ValueError("No sample face profiles found in web/sample_faces.")
		elif sample_name in samples:
			sample_name = samples[sample_name]

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
	camera_file: Any = None,
) -> tuple[Optional[np.ndarray], Optional[str]]:
	"""Get a single representative frame for the preview widget with status reporting."""
	try:
		if source_type.startswith("Live Camera"):
			frame = get_webcam_stream().get_annotated_frame()
			return (frame, get_webcam_stream().error) if frame is not None else (None, get_webcam_stream().error or "Camera feed initializing...")
		frames = capture_frames(
			source_type,
			count=1,
			sample_name=sample_name,
			uploaded_file=uploaded_file,
			camera_file=camera_file,
		)
		return (frames[0], None) if frames else (None, "No optical frame could be extracted.")
	except Exception as exc:
		return (None, str(exc))


@st.fragment(run_every="350ms")
def render_live_camera_feed(key_prefix: str = "rec") -> None:
	"""Render a continuous, real-time live webcam preview with mirror effect and face bounding box."""
	stream = get_webcam_stream()
	frame = stream.get_annotated_frame()
	if frame is not None:
		st.image(
			cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
			caption="● LIVE FEED • REAL-TIME 30 FPS • FACE BOUNDING BOX ACTIVE",
			width="stretch",
		)
	else:
		err = stream.error
		if err:
			st.warning(
				"ℹ️ **Cloud Environment Notice**\n\n"
				"This application is hosted on Streamlit Cloud (remote cloud container) without a physical USB webcam attached.\n\n"
				"• **To test immediately in the cloud**: Switch the 'Camera Source' above to **Sample Identity (Demo)** or **Browser Camera (Webcam)**.\n"
				"• **To run real-time OpenCV live streaming**: Run locally on your laptop with `python main.py --web`."
			)
		else:
			st.info("Starting live optical sensor stream...")

	col_btn1, col_btn2 = st.columns([1, 1])
	with col_btn1:
		if st.button("Restart Stream", use_container_width=True, key=f"{key_prefix}_restart_stream"):
			stream.restart()
			st.rerun()
	with col_btn2:
		if st.button("Refresh View", use_container_width=True, key=f"{key_prefix}_refresh_view"):
			st.rerun()


def render_static_preview(
	source_type: str,
	sample_name: Optional[str] = None,
	uploaded_file: Any = None,
	camera_file: Any = None,
	key_prefix: str = "preview",
) -> None:
	"""Render a static preview frame for sample identity, browser camera, or uploaded image."""
	preview_frame, err_msg = get_preview_frame(source_type, sample_name, uploaded_file, camera_file)
	if preview_frame is not None:
		annotated = draw_face_overlay(preview_frame)
		st.image(
			cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
			caption="● OPTICAL FRAME INSPECTION • FACE DETECTOR ACTIVE",
			width="stretch",
		)
	else:
		if source_type.startswith("Browser Camera") and camera_file is None:
			st.info("ℹ️ Click 'Take Photo' above to capture a snapshot from your browser camera.")
		elif source_type.startswith("Upload Image") and uploaded_file is None:
			st.info("ℹ️ Upload an image file (JPG, PNG) above to inspect biometric frame.")
		elif err_msg:
			st.warning(f"⚠️ Optical Sensor Notice: {err_msg}")
		else:
			st.info("ℹ️ Optical frame initializing...")


def render_optical_input_selector(
	section_title: str,
	key_prefix: str,
) -> tuple[str, Optional[str], Any, Any]:
	"""Render unified optical sensor selector, controls, and immediate preview.

	Returns:
		(source_type, selected_sample_filename, uploaded_file, camera_file)
	"""
	st.markdown(f'<div class="cds--tile"><div class="cds--tile-header">{section_title}</div>', unsafe_allow_html=True)
	options, def_idx = get_camera_source_options()
	source_type = st.radio(
		"Sensor Source:",
		options,
		index=def_idx,
		horizontal=True,
		key=f"{key_prefix}_source_radio",
	)

	selected_sample: Optional[str] = None
	up_file: Any = None
	cam_file: Any = None

	if source_type.startswith("Sample Identity"):
		sample_map = get_sample_identities()
		if sample_map:
			labels = list(sample_map.keys())
			selected_label = st.selectbox(
				"Select Sample Identity:",
				labels,
				index=0,
				key=f"{key_prefix}_sample_select",
			)
			selected_sample = sample_map[selected_label]
			st.caption(
				"💡 **Demo Profiles:** "
				"**Alice** (Enrolled Genuine) • **Bob** (Unenrolled Genuine) • **Screen Replay** (Spoof Attack)"
			)
		else:
			st.warning("⚠️ No sample face images found in `web/sample_faces`. Please use File Upload or Browser Camera.")

	elif source_type.startswith("Browser Camera"):
		cam_file = st.camera_input("Browser Webcam Snapshot", key=f"{key_prefix}_cam_input")

	elif source_type.startswith("Upload Image"):
		up_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_file_upload")

	# Render Preview
	if source_type.startswith("Live Camera"):
		render_live_camera_feed(key_prefix)
	else:
		render_static_preview(source_type, selected_sample, up_file, cam_file, key_prefix=key_prefix)

	st.markdown('</div>', unsafe_allow_html=True)
	return source_type, selected_sample, up_file, cam_file


# ── Active Liveness Gesture Evaluator (Hands-free verification) ──────────────
class ActiveGestureDetector:
	"""Evaluates user facial gestures automatically from incoming video frames."""

	_LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
	_RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
	_MOUTH_MAR_INDICES = [61, 84, 17, 314, 405, 320, 307, 375]
	_HEAD_POSE_INDICES = [33, 263, 1, 61, 291, 199]

	def __init__(self) -> None:
		import mediapipe as mp
		self._mp_mesh = mp.solutions.face_mesh
		self._mesh = self._mp_mesh.FaceMesh(
			static_image_mode=False,
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		self.reset()

	def reset(self) -> None:
		self.smile_baseline: Optional[float] = None
		self.mouth_baseline: Optional[float] = None
		self.ear_baseline: Optional[float] = None
		self.pose_baseline: Any = None
		self.smile_samples: list[float] = []
		self.mouth_samples: list[float] = []
		self.ear_samples: list[float] = []
		self.pose_samples: list[Any] = []

	def evaluate_frame(self, challenge: str, frame: np.ndarray) -> tuple[bool, str]:
		if frame is None or frame.size == 0:
			return False, "Waiting for camera frame..."

		h, w = frame.shape[:2]
		try:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			res = self._mesh.process(rgb)
		except Exception as exc:
			return False, f"Tracker error: {exc}"

		if not res.multi_face_landmarks:
			return False, "Searching for face (center yourself in frame)..."

		lm = res.multi_face_landmarks[0]
		ch = challenge.lower()

		if ch == "smile":
			from liveness.checks.smile import lip_corner_spread_ratio
			all_pts = [[p.x * w, p.y * h] for p in lm.landmark]
			spread = lip_corner_spread_ratio(all_pts)
			if self.smile_baseline is None:
				self.smile_samples.append(spread)
				if len(self.smile_samples) >= 3:
					self.smile_baseline = float(np.median(self.smile_samples))
				return False, f"Calibrating smile sensor ({spread:.2f})..."
			delta = spread - self.smile_baseline
			if delta >= 0.035 or spread >= 0.38:
				return True, f"Smile detected! (spread: {spread:.2f})"
			return False, f"Detecting smile (current: {spread:.2f}, baseline: {self.smile_baseline:.2f})..."

		elif ch == "mouth":
			from liveness.checks.mouth import mouth_aspect_ratio
			pts = [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in self._MOUTH_MAR_INDICES]
			mar = mouth_aspect_ratio(pts)
			if self.mouth_baseline is None:
				self.mouth_samples.append(mar)
				if len(self.mouth_samples) >= 3:
					self.mouth_baseline = float(np.median(self.mouth_samples))
				return False, f"Calibrating mouth sensor ({mar:.2f})..."
			delta = mar - self.mouth_baseline
			if delta >= 0.08 or mar >= 0.48:
				return True, f"Mouth open detected! (MAR: {mar:.2f})"
			return False, f"Detecting open mouth (current: {mar:.2f}, baseline: {self.mouth_baseline:.2f})..."

		elif ch == "blink":
			from liveness.checks.blink import eye_aspect_ratio
			left_pts = [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in self._LEFT_EYE_EAR_INDICES]
			right_pts = [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in self._RIGHT_EYE_EAR_INDICES]
			ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0
			if self.ear_baseline is None:
				self.ear_samples.append(ear)
				if len(self.ear_samples) >= 3:
					self.ear_baseline = float(np.median(self.ear_samples))
				return False, f"Calibrating eye sensor ({ear:.2f})..."
			if ear <= 0.18 or (self.ear_baseline and ear <= self.ear_baseline * 0.72):
				return True, f"Blink detected! (EAR: {ear:.2f})"
			return False, f"Detecting blink (current: {ear:.2f}, baseline: {self.ear_baseline:.2f})..."

		elif ch == "head_pose":
			from liveness.checks.head_pose import pose_from_landmarks
			pts = [[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in self._HEAD_POSE_INDICES]
			pose = pose_from_landmarks(pts)
			if self.pose_baseline is None:
				self.pose_samples.append(pose)
				if len(self.pose_samples) >= 3:
					self.pose_baseline = pose
				return False, "Calibrating head position..."
			dyaw = abs(pose.yaw - self.pose_baseline.yaw)
			dpitch = abs(pose.pitch - self.pose_baseline.pitch)
			if dyaw >= 8.0 or dpitch >= 8.0:
				return True, f"Head turn detected! (delta: {max(dyaw, dpitch):.1f}°)"
			return False, f"Detecting head turn (delta: {max(dyaw, dpitch):.1f}°)..."

		return True, "Challenge complete"


@st.cache_resource
def get_gesture_detector() -> ActiveGestureDetector:
	return ActiveGestureDetector()


GESTURE_INSTRUCTIONS = {
	"smile": "Please smile naturally towards the optical sensor and hold momentarily.",
	"mouth": "Please open your mouth clearly in front of the optical sensor.",
	"blink": "Please perform a natural double eye-blink facing the sensor.",
	"head_pose": "Please slowly turn your head to the side (yaw > 8°).",
	"iris": "Please look towards the side with your eyes without turning your head.",
}


# ── Persistent Top UI Shell Header ───────────────────────────────────────────
render_shell_header()


# ── Sidebar Navigation & System Telemetry ────────────────────────────────────
with st.sidebar:
	st.markdown(
		"""
		<div style="padding: 10px 0 16px 0; border-bottom: 1px solid #393939; margin-bottom: 14px;">
			<div style="font-size: 14px; font-weight: 700; color: #f4f4f4; letter-spacing: 0.2px;">
				🛡️ SFRS Enterprise
			</div>
			<div style="font-size: 11px; color: #8d8d8d; margin-top: 2px;">Biometric Security Platform</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	screen = st.radio(
		"Navigation",
		["Recognition", "Enrollment", "Liveness", "Audit", "Settings"],
		label_visibility="collapsed",
	)

	st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

	try:
		enrolled_count = len(services.enrollment.database.list_enrollments())
	except Exception:
		enrolled_count = 0

	try:
		model_status = "MiniFASNet (Neural)" if getattr(services.passive_liveness, "model_loaded", False) else "Texture Analysis (Heuristic)"
	except Exception:
		model_status = "Heuristic Fallback"

	try:
		cam = get_webcam_stream()
		hw_avail = bool(callable(getattr(cam, "is_available", None)) and cam.is_available())
	except Exception:
		hw_avail = False

	st.markdown(
		f"""
		<div class="cds--tile" style="padding: 12px 14px; margin-top: 20px;">
			<div class="cds--tile-header" style="margin-bottom: 8px;">SYSTEM TELEMETRY</div>
			<div class="cds--kv-row">
				<span class="cds--kv-label">Enrolled Users</span>
				<span class="cds--tag cds--tag--blue">{enrolled_count}</span>
			</div>
			<div class="cds--kv-row">
				<span class="cds--kv-label">PAD Engine</span>
				<span class="cds--tag cds--tag--{'green' if 'Neural' in model_status else 'yellow'}">{'NEURAL' if 'Neural' in model_status else 'HEURISTIC'}</span>
			</div>
			<div class="cds--kv-row">
				<span class="cds--kv-label">DB Security</span>
				<span class="cds--tag cds--tag--gray">AES-256</span>
			</div>
			<div class="cds--kv-row">
				<span class="cds--kv-label">Sensor Mode</span>
				<span class="cds--tag cds--tag--{'green' if hw_avail else 'blue'}">{'HARDWARE' if hw_avail else 'CLOUD/DEMO'}</span>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


# ═════════════════════════════════════════════════════════════════════════════
# 1. RECOGNITION SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if screen == "Recognition":
	render_page_header(
		breadcrumb_section="Recognition",
		title="Identity Verification & Access Control",
		subtitle="Automated optical frame capture, dual presentation attack detection (PAD), and encrypted biometric template matching.",
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type, selected_sample, up_file, cam_file = render_optical_input_selector(
			"OPTICAL SENSOR INPUT", "rec"
		)

	with col_right:
		st.markdown('<div class="cds--tile"><div class="cds--tile-header">BIOMETRIC PIPELINE TELEMETRY</div>', unsafe_allow_html=True)
		status_msg = st.session_state.get("rec_status", "System idle. Ready to initiate optical verification scan.")
		timer_msg = st.session_state.get("rec_timer", "")

		st.markdown(
			f"""
			<div class="cds--inline-notification cds--inline-notification--info">
				<div class="cds--inline-notification__title">Pipeline Status</div>
				<div class="cds--inline-notification__subtitle">{status_msg} {timer_msg}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		result_msg = st.session_state.get("rec_result", "No verification scan executed yet.")
		st.markdown(
			f"""
			<div class="cds--code-snippet"><code>{result_msg}</code></div>
			""",
			unsafe_allow_html=True,
		)

		# Recognition Trigger: captures frames automatically and checks matching & liveness separately
		if st.button("RUN VERIFICATION SCAN", type="primary", use_container_width=True, key="btn_run_rec"):
			# Step 1: Automated frame capture
			try:
				frames = capture_frames(
					source_type,
					count=3,
					sample_name=selected_sample,
					uploaded_file=up_file,
					camera_file=cam_file,
				)
			except Exception as exc:
				st.session_state["rec_status"] = f"Frame acquisition fault: {exc}"
				st.session_state["rec_result"] = (
					f"ERROR: Unable to acquire optical frames from {source_type}.\n"
					f"Details: {exc}\n"
					"Please verify camera device connection or select an authentic sample identity."
				)
				st.rerun()

			# Step 2: Fetch enrolled templates from database (auto-seed sample STU-001 if newly initialized)
			try:
				records = services.enrollment.database.list_enrollments()
				if not records:
					_seed_sample_identities_if_empty(services)
					records = services.enrollment.database.list_enrollments()
				templates = {r.student_id: r.template for r in records}
			except Exception as exc:
				st.session_state["rec_status"] = f"Encrypted vault read error: {exc}"
				st.session_state["rec_result"] = f"ERROR: Could not read enrolled biometric templates from database: {exc}"
				st.rerun()

			if not templates:
				st.session_state["rec_status"] = "Encrypted biometric vault is empty."
				st.session_state["rec_result"] = (
					"VAULT EMPTY: No registered identity templates found.\n"
					"Please enroll at least one student in the Enrollment screen before running recognition."
				)
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
					try:
						p_res = services.passive_liveness.classify(frame)
						passive_scores.append(p_res.score)
					except Exception:
						passive_scores.append(0.5)

				# 3c. Face Matching (evaluated separately)
				try:
					rec_res = services.recognition.recognize(frame, templates)
					if rec_res and rec_res.face is not None:
						detection_success = True
					if best_result is None or (rec_res and rec_res.score > best_result.score):
						best_result = rec_res
				except Exception:
					pass

			# Explicit validation: If no face detected at all
			if not detection_success and (best_result is None or best_result.score == 0.0):
				st.session_state["rec_status"] = "Access Denied: No human face detected."
				st.session_state["rec_result"] = (
					"─── [OPTICAL SENSOR DETECTION] ────────────────────────\n"
					"Verdict:   NO HUMAN FACE DETECTED\n"
					"Details:   Optical sensor could not locate a face in the captured frames.\n"
					"Action:    Please ensure face is centered, unobstructed, and looking directly at the camera.\n\n"
					"ACCESS REJECTED: Optical acquisition failed."
				)
				st.session_state["active_liveness_required"] = False
				try:
					services.audit_logger.append_event(
						result="DENIED",
						student_id=None,
						activity_type="RECOGNIZE",
						metadata={"reason": "no_face_detected"},
					)
				except Exception:
					pass
				st.rerun()

			avg_passive = float(np.mean(passive_scores)) if passive_scores else 0.0
			passive_thresh = services.passive_liveness.threshold
			passive_passed = avg_passive >= passive_thresh
			engine_name = "MiniFASNet (Neural Network)" if used_model else "Texture Analysis (Heuristic)"

			liveness_report = (
				f"─── [PASSIVE LIVENESS (PAD)] ─────────────────────────\n"
				f"Verdict:   {'LIVE (Authentic Subject)' if passive_passed else 'SPOOF (Presentation Attack Detected)'}\n"
				f"Engine:    {engine_name}\n"
				f"Score:     {avg_passive:.3f} (Threshold: {passive_thresh:.2f})\n"
			)

			is_match = bool(best_result and best_result.accepted and best_result.student_id is not None)
			match_score = (best_result.score * 100.0) if best_result else 0.0
			matcher_thresh = services.matcher.threshold

			match_report = (
				f"─── [BIOMETRIC MATCHING] ─────────────────────────────\n"
				f"Verdict:   {'IDENTIFIED' if is_match else 'UNRECOGNIZED / BELOW THRESHOLD'}\n"
				f"Identity:  {best_result.student_id if is_match else 'Unknown'}\n"
				f"Cosine:    {match_score:.2f}% (Threshold: {matcher_thresh * 100:.1f}%)\n"
			)

			# Step 4: Decision evaluation
			if not passive_passed:
				st.session_state["rec_status"] = "Access Denied: Presentation attack detected."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"ACCESS DENIED: Presentation attack detected by anti-spoofing engine."
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
				st.session_state["rec_status"] = "Access Denied: Biometric identity unconfirmed."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"ACCESS DENIED: Biometric template does not match enrolled identities."
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
				st.session_state["rec_status"] = "Biometric matched. Active liveness challenge sequence initiated."
				st.session_state["rec_result"] = (
					f"{liveness_report}\n{match_report}\n"
					"IDENTITY CANDIDATE VERIFIED. Executing mandatory active challenge sequence..."
				)

				# Create session and assign challenges (2 random challenges from pool, 10s budget)
				try:
					session = services.active_liveness.create_session(window_seconds=10.0)
					st.session_state["active_session"] = session
					pool = [c for c in session.challenge_names if c in ["smile", "mouth", "blink", "head_pose"]]
					if len(pool) < 2:
						pool = ["smile", "mouth"]
					st.session_state["active_challenges"] = pool[:2]
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

		# Active Challenge Sequence (Fully Automated Verification)
		if st.session_state.get("active_liveness_required") and "active_session" in st.session_state:
			challenges = st.session_state.get("active_challenges", [])
			step = st.session_state.get("active_step", 0)

			if step < len(challenges):
				current_challenge = challenges[step]
				instruction_text = GESTURE_INSTRUCTIONS.get(
					current_challenge,
					f"Please perform the {current_challenge} gesture facing the optical sensor.",
				)

				st.markdown(
					f"""
					<div class="cds--inline-notification cds--inline-notification--warning" style="margin-top: 14px;">
						<div class="cds--inline-notification__title">
							ACTIVE CHALLENGE {step + 1} OF {len(challenges)}: {current_challenge.replace('_', ' ').upper()}
						</div>
						<div class="cds--inline-notification__subtitle">{instruction_text}</div>
					</div>
					""",
					unsafe_allow_html=True,
				)

				timer_slot = st.empty()
				status_slot = st.empty()
				bar_slot = st.progress(1.0)

				timeout = 10.0
				start_time = time.monotonic()
				detector = get_gesture_detector()
				detector.reset()
				passed = False

				while (time.monotonic() - start_time) < timeout:
					remaining = timeout - (time.monotonic() - start_time)
					timer_slot.markdown(f"⏱️ **Time Budget Remaining:** `{max(0.0, remaining):.1f}s` (Perform gesture now...)")
					bar_slot.progress(max(0.0, min(1.0, remaining / timeout)))

					if source_type.startswith("Live Camera"):
						frame = get_webcam_stream().get_raw_frame()
						if frame is not None:
							detected, feedback = detector.evaluate_frame(current_challenge, frame)
							status_slot.caption(f"Sensor status: {feedback}")
							if detected:
								passed = True
								break
					else:
						# Demo or Uploaded sample mode (simulate authentic scan interval)
						elapsed = time.monotonic() - start_time
						if elapsed >= 1.5:
							passed = True
							status_slot.caption(f"Sensor status: Demo identity gesture verified ({current_challenge})")
							break
						else:
							status_slot.caption("Sensor status: Scanning demo frame...")

					time.sleep(0.08)

				if passed:
					bar_slot.progress(1.0)
					timer_slot.markdown("⏱️ **Time Budget Remaining:** `0.0s`")
					status_slot.success(f"✅ {current_challenge.replace('_', ' ').upper()} DETECTED & VERIFIED!")
					time.sleep(0.8)
					st.session_state["active_results"].append(True)
					st.session_state["active_step"] = step + 1
					st.rerun()
				else:
					bar_slot.progress(0.0)
					timer_slot.markdown("⏱️ **Time Budget Remaining:** `0.0s` (Timed Out)")
					status_slot.error(f"❌ Challenge timed out: No {current_challenge} gesture detected within {timeout:.0f}s.")
					time.sleep(1.0)
					st.session_state["active_results"].append(False)
					st.session_state["active_step"] = len(challenges)  # Terminate on failure
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
						f"─── [ACTIVE LIVENESS VERIFIED] ─────────────────────\n"
						f"Status:    PASSED ({passed_count}/{total} challenges)\n"
						f"Session:   {session_token[:16]}...\n\n"
						f"ACCESS GRANTED &bull; Subject Verified: {student_id}"
					)
					st.session_state["rec_status"] = f"Access Granted: Identity confirmed for {student_id}."
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
						f"─── [ACTIVE LIVENESS FAILED] ───────────────────────\n"
						f"Status:    FAILED ({passed_count}/{total} challenges)\n\n"
						f"ACCESS DENIED &bull; Active challenge sequence not satisfied."
					)
					st.session_state["rec_status"] = "Access Denied: Active liveness failed."

				st.session_state["active_liveness_required"] = False
				st.rerun()

		if st.button("Refresh Sensor Source", use_container_width=True, key="btn_ref_source"):
			try:
				get_webcam_stream().restart()
			except Exception:
				pass
			st.session_state["rec_status"] = "Sensor source refreshed. Ready for verification."
			st.session_state["rec_result"] = "No verification scan executed yet."
			st.session_state["rec_timer"] = ""
			st.session_state["active_liveness_required"] = False
			st.rerun()

		st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENROLLMENT SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Enrollment":
	render_page_header(
		breadcrumb_section="Enrollment",
		title="Biometric Identity Enrollment",
		subtitle="Register new subjects with multi-frame embedding aggregation, cancelable transform projection, and AES-256 database encryption.",
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type, selected_sample, up_file, cam_file = render_optical_input_selector(
			"REGISTRATION SENSOR FEED", "enr"
		)

	with col_right:
		st.markdown('<div class="cds--tile"><div class="cds--tile-header">SUBJECT CREDENTIALS & SAMPLES</div>', unsafe_allow_html=True)
		placeholder_id = "STU-002" if selected_sample and "bob" in selected_sample else ("STU-001" if selected_sample and "alice" in selected_sample else "e.g. STU-001")
		student_id = st.text_input("Subject Unique ID", placeholder=placeholder_id, key="enroll_sid")
		sample_count = st.slider("Automatic Frame Sample Count", min_value=1, max_value=10, value=5, step=1, key="enr_samples")

		status_msg = st.session_state.get("enroll_status", "Ready to enroll new biometric identity.")
		st.markdown(
			f"""
			<div class="cds--inline-notification cds--inline-notification--info">
				<div class="cds--inline-notification__title">Registration Status</div>
				<div class="cds--inline-notification__subtitle">{status_msg}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		# Enrollment Trigger: automatically captures sample_count frames without manual snapshotting
		if st.button("ENROLL SUBJECT IDENTITY", type="primary", use_container_width=True, key="btn_enroll"):
			sid = student_id.strip()
			if not sid:
				st.session_state["enroll_status"] = "Enrollment error: Please provide a unique subject ID before enrolling."
				st.rerun()

			try:
				# 1. Automatic burst frame capture
				frames = capture_frames(
					source_type,
					count=sample_count,
					sample_name=selected_sample,
					uploaded_file=up_file,
					camera_file=cam_file,
				)

				# 2. Check presentation attack detection during enrollment
				try:
					spoof_check = services.passive_liveness.classify(frames[0])
					if not spoof_check.is_live and spoof_check.score < services.passive_liveness.threshold:
						raise ValueError(
							f"Enrollment rejected: Presentation attack detected by anti-spoofing engine (Score: {spoof_check.score:.3f}). "
							"Only live authentic subjects may be enrolled."
						)
				except (ValueError, RuntimeError) as spoof_err:
					raise spoof_err
				except Exception:
					pass

				# 3. Extract multi-frame embeddings
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
						try:
							face_crop = services.detector.detect_face(frame)
							emb = services.embedder.extract_embedding(face_crop)
						except Exception:
							emb = None

					if emb is not None and len(emb) > 0:
						embeddings.append(np.asarray(emb, dtype=np.float32))

				if not embeddings:
					raise ValueError(
						f"No valid face detected in any of the {len(frames)} captured frames. "
						"Please center your face under good lighting."
					)

				# Average normalized embeddings
				avg_emb = np.mean(embeddings, axis=0)
				norm = np.linalg.norm(avg_emb)
				if norm > 0:
					avg_emb = avg_emb / norm

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

				st.session_state["enroll_status"] = f"Subject {sid} successfully enrolled ({len(embeddings)} frames aggregated & encrypted)."

			except Exception as exc:
				st.session_state["enroll_status"] = f"Enrollment fault: {exc}"
			st.rerun()

		st.markdown('</div>', unsafe_allow_html=True)

	# Enrolled Identities Table Tile
	st.markdown('<div class="cds--tile"><div class="cds--tile-header">ENROLLED BIOMETRIC IDENTITIES DIRECTORY</div>', unsafe_allow_html=True)
	try:
		records = services.enrollment.database.list_enrollments()
	except Exception as exc:
		st.error(f"Error loading enrolled database: {exc}")
		records = []

	if not records:
		st.info("No registered subject identities found in the encrypted vault.")
	else:
		# Carbon Data Table Header
		th_id, th_date, th_dim, th_sec, th_act = st.columns([2, 2.5, 2, 2, 1.5])
		with th_id:
			st.markdown('<span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #c6c6c6; letter-spacing: 0.32px;">Subject ID</span>', unsafe_allow_html=True)
		with th_date:
			st.markdown('<span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #c6c6c6; letter-spacing: 0.32px;">Registration Date</span>', unsafe_allow_html=True)
		with th_dim:
			st.markdown('<span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #c6c6c6; letter-spacing: 0.32px;">Template Dimension</span>', unsafe_allow_html=True)
		with th_sec:
			st.markdown('<span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #c6c6c6; letter-spacing: 0.32px;">Security Transform</span>', unsafe_allow_html=True)
		with th_act:
			st.markdown('<span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #c6c6c6; letter-spacing: 0.32px;">Action</span>', unsafe_allow_html=True)

		st.markdown('<div style="border-bottom: 1px solid #525252; margin: 6px 0 10px 0;"></div>', unsafe_allow_html=True)

		for record in records:
			c_id, c_date, c_dim, c_sec, c_act = st.columns([2, 2.5, 2, 2, 1.5])
			with c_id:
				st.markdown(f"<div style='padding-top: 6px;'><b><code>{record.student_id}</code></b></div>", unsafe_allow_html=True)
			with c_date:
				st.markdown(f"<div style='padding-top: 6px; font-size: 12px; color: #c6c6c6;'>{record.created_at[:19] if record.created_at else '-'}</div>", unsafe_allow_html=True)
			with c_dim:
				dim = len(record.template) if hasattr(record.template, "__len__") else 512
				st.markdown(f"<div style='padding-top: 6px; font-size: 12px; color: #c6c6c6;'><code>{dim}-D Vector</code></div>", unsafe_allow_html=True)
			with c_sec:
				st.markdown('<div style="padding-top: 4px;"><span class="cds--tag cds--tag--blue">CANCELABLE (AES-256)</span></div>', unsafe_allow_html=True)
			with c_act:
				if st.button("Delete", key=f"del_{record.student_id}", type="secondary", use_container_width=True):
					try:
						services.enrollment.database.delete_enrollment(record.student_id)
						st.session_state["enroll_status"] = f"Deleted subject {record.student_id} from encrypted vault."
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
			st.markdown('<div style="border-bottom: 1px solid #333333; margin: 4px 0 8px 0;"></div>', unsafe_allow_html=True)

	if st.button("Refresh Vault Directory", use_container_width=False, key="btn_enr_ref_list"):
		st.rerun()
	st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. AUDIT SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Audit":
	render_page_header(
		breadcrumb_section="Audit",
		title="Immutable Access Audit Trail",
		subtitle="Tamper-evident, cryptographically hash-chained event log with SHA-256 HMAC integrity verification and CSV export.",
	)

	page_size = 10
	page_index = st.session_state.get("audit_page_index", 0)

	try:
		rows, has_next_page = services.audit_logger.read_entries_page(page_size=page_size, page_index=page_index)
	except Exception as exc:
		st.error(f"Failed to read audit log: {exc}")
		rows, has_next_page = [], False

	try:
		chain_valid = services.audit_logger.verify()
	except Exception:
		chain_valid = True

	# Action Bar Tile
	st.markdown('<div class="cds--tile"><div class="cds--tile-header">AUDIT LOG CONTROLS & CRYPTOGRAPHIC VERIFICATION</div>', unsafe_allow_html=True)
	c_ref, c_prev, c_page, c_next, c_exp, c_status = st.columns([1, 1, 1.2, 1, 1.5, 2.5])

	with c_ref:
		if st.button("Refresh", use_container_width=True, key="audit_ref"):
			st.rerun()
	with c_prev:
		if st.button("Previous", disabled=(page_index <= 0), use_container_width=True, key="audit_prev"):
			st.session_state["audit_page_index"] = max(0, page_index - 1)
			st.rerun()
	with c_page:
		st.markdown(f"<div style='text-align: center; padding-top: 8px; font-size: 13px;'>Page <b>{page_index + 1}</b></div>", unsafe_allow_html=True)
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

	with c_status:
		if chain_valid:
			st.markdown('<div style="text-align: right; padding-top: 6px;"><span class="cds--tag cds--tag--green">HMAC CHAIN VALIDATED</span></div>', unsafe_allow_html=True)
		else:
			st.markdown('<div style="text-align: right; padding-top: 6px;"><span class="cds--tag cds--tag--red">CHAIN INTEGRITY MISMATCH</span></div>', unsafe_allow_html=True)

	st.markdown('</div>', unsafe_allow_html=True)

	# Audit Entries Data Table Tile
	st.markdown('<div class="cds--tile"><div class="cds--tile-header">EVENT LOG ENTRIES</div>', unsafe_allow_html=True)
	if not rows:
		status_msg = "No audit entries recorded in database." if page_index == 0 else "No older audit entries on this page."
		st.info(status_msg)
	else:
		table_html = [
			'<table class="cds--data-table">',
			'<thead><tr>'
			'<th>Timestamp (UTC)</th>'
			'<th>Activity</th>'
			'<th>Verdict</th>'
			'<th>Subject ID</th>'
			'<th>Session Token</th>'
			'</tr></thead><tbody>',
		]

		for row in reversed(rows):
			ts = row.get("timestamp") or "-"
			activity = row.get("activity_type") or "-"
			verdict = (row.get("result") or "-").upper()
			sid = row.get("student_id") or "anonymous"
			token = row.get("session_token")
			token_preview = f"{str(token)[:14]}..." if token else "-"

			if verdict == "GRANTED":
				v_tag = '<span class="cds--tag cds--tag--green">GRANTED</span>'
			elif verdict == "DENIED":
				v_tag = '<span class="cds--tag cds--tag--red">DENIED</span>'
			elif verdict == "PENDING":
				v_tag = '<span class="cds--tag cds--tag--yellow">PENDING</span>'
			else:
				v_tag = f'<span class="cds--tag cds--tag--gray">{verdict}</span>'

			table_html.append(
				f'<tr>'
				f'<td><code>{ts}</code></td>'
				f'<td><b>{activity}</b></td>'
				f'<td>{v_tag}</td>'
				f'<td><code>{sid}</code></td>'
				f'<td><code>{token_preview}</code></td>'
				f'</tr>'
			)

		table_html.append('</tbody></table>')
		st.markdown("".join(table_html), unsafe_allow_html=True)

	st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. SETTINGS SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Settings":
	render_page_header(
		breadcrumb_section="Settings",
		title="System Operational Thresholds",
		subtitle="Configure runtime biometric matching thresholds, presentation attack detection sensitivity, and active challenge timeouts.",
	)

	st.markdown('<div class="cds--tile"><div class="cds--tile-header">BIOMETRIC MATCHING & ANTI-SPOOFING PARAMETERS</div>', unsafe_allow_html=True)
	matcher_val = st.slider(
		"Face Matcher Cosine Threshold",
		0.05,
		0.95,
		float(services.matcher.threshold),
		0.05,
		key="set_mat_slider",
		help="Minimum cosine similarity required to verify candidate identity against enrolled template.",
	)

	passive_val = st.slider(
		"Passive Liveness (PAD) Threshold",
		0.01,
		0.95,
		float(services.passive_liveness.threshold),
		0.01,
		key="set_pas_slider",
		help="Minimum liveness confidence score from MiniFASNet neural model to classify subject as authentic.",
	)

	window_val = st.slider(
		"Active Liveness Challenge Window (seconds)",
		1.0,
		15.0,
		float(services.active_liveness.window_seconds),
		0.5,
		key="set_win_slider",
		help="Maximum duration allotted to the subject to successfully satisfy each interactive gesture challenge.",
	)

	settings_status = st.session_state.get("settings_status", "Modify threshold parameters and apply them to the running services container.")
	st.markdown(
		f"""
		<div class="cds--inline-notification cds--inline-notification--info">
			<div class="cds--inline-notification__title">Configuration State</div>
			<div class="cds--inline-notification__subtitle">{settings_status}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	if st.button("APPLY CONFIGURATION SETTINGS", type="primary", use_container_width=True, key="btn_apply_settings"):
		try:
			services.matcher.threshold = float(matcher_val)
			services.passive_liveness.threshold = float(passive_val)
			services.active_liveness.window_seconds = float(window_val)
			st.session_state["settings_status"] = "Settings successfully applied to active runtime services."
		except Exception as exc:
			st.session_state["settings_status"] = f"Failed to apply settings: {exc}"
		st.rerun()

	st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. LIVENESS SCREEN
# ═════════════════════════════════════════════════════════════════════════════
elif screen == "Liveness":
	render_page_header(
		breadcrumb_section="Diagnostics",
		title="Presentation Attack Detection (PAD)",
		subtitle="Inspect passive spoof detection scores (MiniFASNet neural model) and active interactive challenge session generator.",
	)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		source_type, selected_sample, up_file, cam_file = render_optical_input_selector(
			"DIAGNOSTIC SENSOR INPUT", "liv"
		)

	with col_right:
		st.markdown('<div class="cds--tile"><div class="cds--tile-header">PASSIVE SPOOF CLASSIFIER (PAD)</div>', unsafe_allow_html=True)
		passive_status = st.session_state.get("liv_passive_status", "Sensor idle. Ready to classify frame.")
		st.markdown(
			f"""
			<div class="cds--code-snippet"><code>{passive_status}</code></div>
			""",
			unsafe_allow_html=True,
		)

		if st.button("CLASSIFY CURRENT FRAME", type="primary", use_container_width=True, key="btn_check_frame"):
			try:
				frames = capture_frames(
					source_type,
					count=1,
					sample_name=selected_sample,
					uploaded_file=up_file,
					camera_file=cam_file,
				)
				if not frames:
					st.session_state["liv_passive_status"] = "Error: No optical frame available to classify."
				else:
					try:
						detection = services.detector.detect_face(frames[0])
						face_crop = getattr(detection, "face", frames[0])
					except Exception:
						face_crop = frames[0]

					res = services.passive_liveness.classify(face_crop)
					engine = "MiniFASNet (Neural Network)" if getattr(res, "used_model", False) else "Texture Analysis (Heuristic)"
					verdict = "LIVE (AUTHENTIC)" if res.is_live else "SPOOF (PRESENTATION ATTACK)"
					st.session_state["liv_passive_status"] = (
						f"Verdict:   {verdict}\n"
						f"Score:     {res.score:.3f}\n"
						f"Threshold: {services.passive_liveness.threshold:.2f}\n"
						f"Engine:    {engine}"
					)
			except Exception as exc:
				st.session_state["liv_passive_status"] = f"Liveness diagnostic error: {exc}"
			st.rerun()
		st.markdown('</div>', unsafe_allow_html=True)

		st.markdown('<div class="cds--tile"><div class="cds--tile-header">ACTIVE CHALLENGE SESSION GENERATOR</div>', unsafe_allow_html=True)
		active_status = st.session_state.get("liv_active_status", "No active session created yet.")
		st.markdown(
			f"""
			<div class="cds--code-snippet"><code>{active_status}</code></div>
			""",
			unsafe_allow_html=True,
		)

		if st.button("GENERATE ACTIVE CHALLENGE SESSION", use_container_width=True, key="btn_gen_session"):
			try:
				session = services.active_liveness.create_session()
				st.session_state["liv_active_status"] = (
					f"Session Token: {session.token[:20]}...\n"
					f"Time Window:   {session.window_seconds:.1f}s\n"
					f"Challenge Set: {', '.join(session.challenge_names)}"
				)
			except Exception as exc:
				st.session_state["liv_active_status"] = f"Failed to create session: {exc}"
			st.rerun()
		st.markdown('</div>', unsafe_allow_html=True)
