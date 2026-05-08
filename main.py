"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import os

from security.runtime_checks import validate_runtime_security


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Secure Face Recognition System")
	parser.add_argument(
		"--validate-security",
		action="store_true",
		help="Run runtime security checks and exit",
	)
	return parser.parse_args()


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

	from ui.app import run_app

	run_app()


if __name__ == "__main__":
	main()

