import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.attack_sim import load_spoof_samples, simulate_attack
from evaluation.metrics import (
    candidate_thresholds,
    compute_eer,
    compute_threshold_metrics,
    roc_auc,
    roc_curve_points,
)
from liveness.passive import PassiveLivenessChecker


class ConstantPassiveModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, _frame):
        return np.array([self.score], dtype=np.float32)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.genuine = [0.92, 0.88, 0.79, 0.84, 0.9]
        self.impostor = [0.12, 0.21, 0.33, 0.4, 0.18]

    def test_compute_threshold_metrics(self) -> None:
        metrics = compute_threshold_metrics(self.genuine, self.impostor, threshold=0.5)
        self.assertEqual(metrics.false_accepts, 0)
        self.assertEqual(metrics.false_rejects, 0)
        self.assertAlmostEqual(metrics.far, 0.0, places=6)
        self.assertAlmostEqual(metrics.frr, 0.0, places=6)

    def test_candidate_thresholds(self) -> None:
        thresholds = candidate_thresholds(self.genuine, self.impostor, steps=50)
        self.assertEqual(thresholds.ndim, 1)
        self.assertGreaterEqual(thresholds.size, 2)

    def test_compute_eer_returns_reasonable_value(self) -> None:
        result = compute_eer(self.genuine, self.impostor, steps=100)
        self.assertGreaterEqual(result.eer, 0.0)
        self.assertLessEqual(result.eer, 1.0)

    def test_roc_points_and_auc(self) -> None:
        fpr, tpr, thresholds = roc_curve_points(self.genuine, self.impostor, steps=100)
        self.assertEqual(fpr.shape, tpr.shape)
        self.assertEqual(fpr.shape, thresholds.shape)
        area = roc_auc(fpr, tpr)
        self.assertGreaterEqual(area, 0.0)
        self.assertLessEqual(area, 1.0)


class TestAttackSimulation(unittest.TestCase):
    def test_load_spoof_samples_from_npy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.npy"
            np.save(path, np.zeros((4, 4, 3), dtype=np.uint8))
            loaded = load_spoof_samples(temp_dir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].shape, (4, 4, 3))

    def test_simulate_attack(self) -> None:
        checker = PassiveLivenessChecker(model=ConstantPassiveModel(0.1), threshold=0.5)
        samples = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        result = simulate_attack(samples, checker)
        self.assertEqual(result.total_samples, 5)
        self.assertEqual(result.detected_spoofs, 5)
        self.assertAlmostEqual(result.detection_rate, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
