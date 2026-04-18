"""Active liveness challenge orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import random
import secrets
import time
from typing import Any, Callable, Iterable, Optional

from liveness.checks.blink import evaluate_blink
from liveness.checks.head_pose import evaluate_head_pose
from liveness.checks.iris import evaluate_iris_tracking
from liveness.checks.mouth import evaluate_mouth_open
from liveness.checks.smile import evaluate_smile


class ActiveLivenessError(RuntimeError):
	"""Raised when active liveness orchestration fails."""


def generate_session_token(timestamp: Optional[str] = None, nonce: Optional[str] = None) -> str:
	stamp = timestamp or datetime.now(timezone.utc).isoformat()
	value = nonce or secrets.token_hex(16)
	return hashlib.sha256(f"{stamp}:{value}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChallengeResult:
	name: str
	passed: bool
	value: float | None = None
	threshold: float | None = None
	details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveLivenessSession:
	token: str
	challenge_names: tuple[str, ...]
	window_seconds: float
	created_at: str


def _challenge_registry() -> dict[str, Callable[..., bool]]:
	return {
		"blink": evaluate_blink,
		"head_pose": evaluate_head_pose,
		"mouth": evaluate_mouth_open,
		"smile": evaluate_smile,
		"iris": evaluate_iris_tracking,
	}


def default_challenge_names() -> list[str]:
	return list(_challenge_registry().keys())


class ActiveLivenessOrchestrator:
	"""Randomly selects and executes active liveness challenges."""

	def __init__(
		self,
		challenge_pool: Optional[Iterable[str]] = None,
		min_challenges: int = 2,
		max_challenges: int = 3,
		window_seconds: float = 2.0,
		rng: Optional[random.Random] = None,
	) -> None:
		self.challenge_pool = tuple(challenge_pool or default_challenge_names())
		self.min_challenges = min_challenges
		self.max_challenges = max_challenges
		self.window_seconds = float(window_seconds)
		self.rng = rng or random.Random()

	def create_session(self) -> ActiveLivenessSession:
		if self.min_challenges <= 0:
			raise ActiveLivenessError("min_challenges must be positive")
		if self.max_challenges < self.min_challenges:
			raise ActiveLivenessError("max_challenges must be >= min_challenges")
		if len(self.challenge_pool) < self.min_challenges:
			raise ActiveLivenessError("challenge pool is too small")

		count = self.rng.randint(self.min_challenges, min(self.max_challenges, len(self.challenge_pool)))
		selected = tuple(self.rng.sample(list(self.challenge_pool), count))
		created_at = datetime.now(timezone.utc).isoformat()
		return ActiveLivenessSession(
			token=generate_session_token(created_at),
			challenge_names=selected,
			window_seconds=self.window_seconds,
			created_at=created_at,
		)

	@staticmethod
	def _check_time_budget(start_time: float, window_seconds: float) -> None:
		elapsed = time.monotonic() - start_time
		if elapsed > window_seconds:
			raise ActiveLivenessError(f"Challenge timed out after {elapsed:.3f}s")

	def _run_session(
		self,
		session: ActiveLivenessSession,
		challenge_inputs: dict[str, Any],
		baseline: Optional[dict[str, Any]] = None,
		thresholds: Optional[dict[str, float]] = None,
	) -> list[ChallengeResult]:
		results: list[ChallengeResult] = []
		start_time = time.monotonic()
		registry = _challenge_registry()
		baseline = baseline or {}
		thresholds = thresholds or {}

		for challenge_name in session.challenge_names:
			self._check_time_budget(start_time, session.window_seconds)
			challenge = registry.get(challenge_name)
			if challenge is None:
				raise ActiveLivenessError(f"Unknown challenge {challenge_name!r}")
			challenge_input = challenge_inputs.get(challenge_name)
			if challenge_input is None:
				raise ActiveLivenessError(f"Missing input for challenge {challenge_name!r}")

			if challenge_name == "blink":
				passed = challenge(challenge_input, baseline=baseline.get("blink"), threshold=thresholds.get("blink"))
			elif challenge_name == "head_pose":
				passed = challenge(challenge_input, baseline=baseline.get("head_pose"), threshold=thresholds.get("head_pose"))
			elif challenge_name == "mouth":
				passed = challenge(challenge_input, threshold=thresholds.get("mouth"))
			elif challenge_name == "smile":
				passed = challenge(challenge_input, threshold=thresholds.get("smile"))
			elif challenge_name == "iris":
				passed = challenge(challenge_input, target=baseline.get("iris"), tolerance=thresholds.get("iris"))
			else:
				passed = bool(challenge(challenge_input))

			results.append(
				ChallengeResult(
					name=challenge_name,
					passed=bool(passed),
					details={"token": session.token},
				)
			)

		return results

	def run(
		self,
		challenge_inputs: dict[str, Any],
		baseline: Optional[dict[str, Any]] = None,
		thresholds: Optional[dict[str, float]] = None,
	) -> list[ChallengeResult]:
		session = self.create_session()
		return self._run_session(
			session=session,
			challenge_inputs=challenge_inputs,
			baseline=baseline,
			thresholds=thresholds,
		)

	def create_and_run(
		self,
		challenge_inputs: dict[str, Any],
		baseline: Optional[dict[str, Any]] = None,
		thresholds: Optional[dict[str, float]] = None,
	) -> tuple[ActiveLivenessSession, list[ChallengeResult]]:
		session = self.create_session()
		results = self._run_session(
			session=session,
			challenge_inputs=challenge_inputs,
			baseline=baseline,
			thresholds=thresholds,
		)
		return session, results

