"""Application entry point supporting Desktop GUI, Web Interface, and Security Preflight."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from security.runtime_checks import validate_runtime_security


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Secure Face Recognition System")
	parser.add_argument(
		"--validate-security",
		action="store_true",
		help="Run runtime security checks and exit",
	)
	parser.add_argument(
		"--web",
		action="store_true",
		help="Launch the interactive Streamlit web interface instead of desktop GUI",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=int(os.getenv("PORT", "7860")),
		help="Port for the web interface (default: 7860 or $PORT)",
	)
	return parser.parse_args()


def run_web_app(port: int = 7860) -> None:
	"""Launch the Streamlit web application."""
	web_app = Path(__file__).parent / "web" / "app.py"
	if not web_app.exists():
		print(f"Error: Web app file not found at {web_app}", file=sys.stderr)
		sys.exit(1)

	cmd = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		str(web_app),
		"--server.port",
		str(port),
		"--server.address",
		"0.0.0.0",
		"--server.headless",
		"true",
	]
	print(f"Starting Secure Face Recognition Web Interface on http://localhost:{port}")
	sys.exit(subprocess.call(cmd))


def main() -> None:
	args = _parse_args()
	if os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
		logging.basicConfig(
			level=logging.INFO,
			format="[%(levelname)s] %(name)s: %(message)s",
		)
		print("[DEBUG] LIVENESS_DEBUG enabled: verbose liveness logs will be printed.")

	if args.validate_security:
		ok, issues = validate_runtime_security()
		if ok:
			print("Runtime security checks: OK")
			return
		print("Runtime security checks: FAILED")
		for issue in issues:
			print(f"- {issue}")
		raise SystemExit(1)

	if args.web:
		run_web_app(port=args.port)
		return

	# If running headless or DISPLAY is not set on Unix, fallback or guide to --web
	if sys.platform != "win32" and not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
		print("[INFO] Headless environment detected (no display server). Launching web interface...")
		run_web_app(port=args.port)
		return

	try:
		from ui.app import run_app
		run_app()
	except Exception as exc:
		print(f"[WARNING] Could not launch desktop GUI ({exc}). Falling back to web interface...")
		run_web_app(port=args.port)


if __name__ == "__main__":
	main()
