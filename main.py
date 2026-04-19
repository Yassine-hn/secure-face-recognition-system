"""Application entry point."""

from __future__ import annotations

import argparse

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

