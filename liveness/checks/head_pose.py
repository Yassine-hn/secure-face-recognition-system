"""Head pose estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class HeadPoseDetectionError(ValueError):
	"""Raised when head pose inputs are malformed."""


@dataclass(frozen=True)
class HeadPose:
	pitch: float
	yaw: float
	roll: float


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
	array = np.asarray(list(points), dtype=np.float32)
	if array.ndim != 2 or array.shape[1] != 2:
		raise HeadPoseDetectionError(f"points must be an Nx2 array, got shape {array.shape}")
	return array


def pose_from_landmarks(points: Iterable[Sequence[float]]) -> HeadPose:
	"""Estimate head pose from a simplified landmark set.

	This is a deterministic fallback used in tests. If cv2.solvePnP is available,
	a higher fidelity estimator can be injected later.
	"""
	landmarks = _as_points(points)
	if landmarks.shape[0] < 6:
		raise HeadPoseDetectionError("at least 6 points are required")

	left_eye = landmarks[0]
	right_eye = landmarks[1]
	nose = landmarks[2]
	left_mouth = landmarks[3]
	right_mouth = landmarks[4]
	chin = landmarks[5]

	face_width = np.linalg.norm(left_eye - right_eye)
	face_height = np.linalg.norm(nose - chin)
	if face_width == 0.0 or face_height == 0.0:
		raise HeadPoseDetectionError("reference distances must not be zero")

	yaw = float((right_eye[0] - left_eye[0]) / face_width * 45.0)
	pitch = float((nose[1] - (left_eye[1] + right_eye[1]) / 2.0) / face_height * 45.0)
	roll = float(((right_mouth[1] - left_mouth[1]) / face_width) * 45.0)
	return HeadPose(pitch=pitch, yaw=yaw, roll=roll)


def relative_pose_delta(current: HeadPose, baseline: HeadPose) -> HeadPose:
	return HeadPose(
		pitch=current.pitch - baseline.pitch,
		yaw=current.yaw - baseline.yaw,
		roll=current.roll - baseline.roll,
	)


def head_pose_within_threshold(delta: HeadPose, threshold: float = 10.0) -> bool:
	if threshold <= 0.0:
		raise HeadPoseDetectionError("threshold must be positive")
	return max(abs(delta.pitch), abs(delta.yaw), abs(delta.roll)) >= threshold


def evaluate_head_pose(
	points: Iterable[Sequence[float]],
	baseline: HeadPose | Sequence[float] | None = None,
	threshold: float | None = None,
) -> bool:
	"""Evaluate a head pose challenge from landmarks.

	The check passes when the relative pose delta reaches the requested threshold.
	"""
	current = pose_from_landmarks(points)
	if isinstance(baseline, HeadPose):
		delta = relative_pose_delta(current, baseline)
	else:
		if baseline is None:
			baseline_pose = HeadPose(0.0, 0.0, 0.0)
		else:
			values = tuple(float(value) for value in baseline)
			if len(values) != 3:
				raise HeadPoseDetectionError("baseline must contain 3 values")
			baseline_pose = HeadPose(*values)
		delta = relative_pose_delta(current, baseline_pose)
	if threshold is None:
		threshold = 10.0
	return head_pose_within_threshold(delta, threshold)

