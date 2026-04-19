"""Recognition backend benchmarking helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from core.matcher import FaceMatcher


@dataclass(frozen=True)
class BackendBenchmarkResult:
	backend: str
	accuracy: float
	avg_latency_ms: float
	samples: int


def benchmark_backends(
	backend_extractors: Mapping[str, Callable[[Any], np.ndarray]],
	samples: Iterable[tuple[Any, str]],
	enrolled_templates: Mapping[str, np.ndarray],
	matcher_threshold: float = 0.5,
) -> list[BackendBenchmarkResult]:
	"""Benchmark multiple embedding extractors over labeled samples."""
	data = list(samples)
	if not data:
		return []

	results: list[BackendBenchmarkResult] = []
	for backend_name, extractor in backend_extractors.items():
		matcher = FaceMatcher(threshold=matcher_threshold)
		correct = 0
		latencies_ms: list[float] = []

		for frame, expected_id in data:
			start = time.perf_counter()
			embedding = np.asarray(extractor(frame), dtype=np.float32)
			latencies_ms.append((time.perf_counter() - start) * 1000.0)

			predicted_id, _score = matcher.identify(embedding, enrolled_templates)
			if predicted_id == expected_id:
				correct += 1

		accuracy = correct / float(len(data))
		avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
		results.append(
			BackendBenchmarkResult(
				backend=backend_name,
				accuracy=accuracy,
				avg_latency_ms=avg_latency,
				samples=len(data),
			)
		)

	results.sort(key=lambda item: (item.accuracy, -item.avg_latency_ms), reverse=True)
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Benchmark recognition backends")
	parser.add_argument("--note", default="Provide custom dataset loader in code", help="Informational note")
	args = parser.parse_args()
	print("Benchmark module is ready.")
	print(args.note)
	print("Use benchmark_backends(...) from Python with your dataset and enrolled templates.")


if __name__ == "__main__":
	main()

