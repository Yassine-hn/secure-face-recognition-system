"""Evaluate passive liveness against spoof sample frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from liveness.passive import PassiveLivenessChecker


@dataclass(frozen=True)
class AttackSimulationResult:
	total_samples: int
	detected_spoofs: int
	detection_rate: float
	false_live: int


def load_spoof_samples(spoof_dir: str | Path) -> list[np.ndarray]:
	"""Load spoof sample frames from a directory.

	Supported formats: .npy always, image files when OpenCV is available.
	"""
	directory = Path(spoof_dir)
	if not directory.exists() or not directory.is_dir():
		raise FileNotFoundError(f"Invalid spoof directory: {directory}")

	samples: list[np.ndarray] = []
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	use_cv2 = False
	try:
		import cv2
		use_cv2 = True
	except Exception:
		cv2 = None  # type: ignore[assignment]

	for path in sorted(directory.iterdir()):
		if path.suffix.lower() == ".npy":
			array = np.load(path)
			samples.append(np.asarray(array))
		elif path.suffix.lower() in image_extensions and use_cv2:
			image = cv2.imread(str(path))  # type: ignore[attr-defined]
			if image is not None:
				samples.append(image)

	return samples


def simulate_attack(
	spoof_samples: Iterable[np.ndarray],
	checker: PassiveLivenessChecker,
) -> AttackSimulationResult:
	"""Run spoof samples through passive liveness and compute detection rate."""
	total = 0
	detected_spoofs = 0
	for sample in spoof_samples:
		total += 1
		result = checker.classify(sample)
		if not result.is_live:
			detected_spoofs += 1
	false_live = total - detected_spoofs
	detection_rate = 0.0 if total == 0 else detected_spoofs / float(total)
	return AttackSimulationResult(
		total_samples=total,
		detected_spoofs=detected_spoofs,
		detection_rate=detection_rate,
		false_live=false_live,
	)


def run_attack_simulation(spoof_dir: str | Path, threshold: float = 0.5) -> AttackSimulationResult:
	samples = load_spoof_samples(spoof_dir)
	checker = PassiveLivenessChecker(threshold=threshold)
	return simulate_attack(samples, checker)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run passive liveness attack simulation")
	parser.add_argument("--spoof-dir", required=True, help="Directory with spoof sample frames")
	parser.add_argument("--threshold", type=float, default=0.5, help="Passive liveness decision threshold")
	args = parser.parse_args()

	result = run_attack_simulation(args.spoof_dir, threshold=args.threshold)
	print(f"Total samples: {result.total_samples}")
	print(f"Detected spoofs: {result.detected_spoofs}")
	print(f"False live: {result.false_live}")
	print(f"Detection rate: {result.detection_rate:.4f}")


if __name__ == "__main__":
	main()

