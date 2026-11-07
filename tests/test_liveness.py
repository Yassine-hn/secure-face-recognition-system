import random
import unittest

import numpy as np

from liveness.active import ActiveLivenessOrchestrator, ActiveLivenessError, generate_session_token
from liveness.checks.blink import BlinkDetectionError, blink_detected, evaluate_blink, eye_aspect_ratio
from liveness.checks.head_pose import HeadPose, HeadPoseDetectionError, evaluate_head_pose, pose_from_landmarks
from liveness.checks.iris import IrisDetectionError, IrisTarget, evaluate_iris_tracking, iris_centroid
from liveness.checks.mouth import evaluate_mouth_open, mouth_aspect_ratio
from liveness.checks.smile import evaluate_smile, lip_corner_spread_ratio
from liveness.passive import PassiveLivenessChecker, PassiveLivenessResult


class FakePassiveModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, face_image):
        return np.array([self.score], dtype=np.float32)


class TestPassiveLiveness(unittest.TestCase):
    def test_classify_uses_injected_model(self) -> None:
        checker = PassiveLivenessChecker(model=FakePassiveModel(0.9), threshold=0.5)
        result = checker.classify(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIsInstance(result, PassiveLivenessResult)
        self.assertTrue(result.is_live)
        self.assertEqual(result.label, "live")

    def test_classify_marks_spoof_below_threshold(self) -> None:
        checker = PassiveLivenessChecker(model=FakePassiveModel(0.1), threshold=0.5)
        result = checker.classify(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertFalse(result.is_live)
        self.assertEqual(result.label, "spoof")

    def test_classify_with_frame_and_bbox(self) -> None:
        class FakeModelWithBBox:
            def predict(self, frame_or_crop, bbox=None):
                if bbox is not None:
                    return np.array([0.05, 0.95, 0.0], dtype=np.float32)
                return np.array([0.5, 0.5, 0.0], dtype=np.float32)

        checker = PassiveLivenessChecker(model=FakeModelWithBBox(), threshold=0.5)
        res = checker.classify(
            np.zeros((10, 10, 3), dtype=np.uint8),
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            bbox=(10, 10, 40, 40),
        )
        self.assertTrue(res.is_live)
        self.assertAlmostEqual(res.score, 0.95, places=4)

    def test_minifasnet_native_architecture(self) -> None:
        import torch
        from liveness.minifasnet_model import MiniFASNetV2, CropImage, get_kernel, parse_model_name

        h, w, mtype, scale = parse_model_name("2.7_80x80_MiniFASNetV2.pth")
        self.assertEqual((h, w), (80, 80))
        self.assertEqual(scale, 2.7)

        kernel = get_kernel(h, w)
        net = MiniFASNetV2(conv6_kernel=kernel)
        dummy_input = torch.zeros((1, 3, 80, 80), dtype=torch.float32)
        net.eval()
        with torch.no_grad():
            out = net(dummy_input)
        self.assertEqual(out.shape, (1, 3))

        # Test CropImage
        cropper = CropImage()
        org = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped = cropper.crop(org, bbox=[50, 50, 40, 40], scale=scale, out_w=80, out_h=80)
        self.assertEqual(cropped.shape, (80, 80, 3))


class TestBlinkChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.eye_points = [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (4.0, 0.0),
            (2.0, -1.0),
            (1.0, -1.0),
        ]

    def test_eye_aspect_ratio(self) -> None:
        self.assertAlmostEqual(eye_aspect_ratio(self.eye_points), 0.5, places=6)

    def test_blink_detected_against_baseline(self) -> None:
        self.assertTrue(blink_detected(0.2, 0.5))
        self.assertFalse(blink_detected(0.45, 0.5))

    def test_evaluate_blink_with_threshold(self) -> None:
        self.assertTrue(evaluate_blink(self.eye_points, threshold=0.6))


class TestMouthAndSmileChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.mouth_points = [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 2.0),
            (4.0, 0.0),
            (3.0, -2.0),
            (2.0, -2.0),
            (1.0, -1.0),
        ]
        self.landmarks = [(float(index), float(index % 5)) for index in range(500)]
        self.landmarks[61] = (10.0, 10.0)
        self.landmarks[291] = (30.0, 10.0)
        self.landmarks[234] = (0.0, 10.0)
        self.landmarks[454] = (40.0, 10.0)

    def test_mouth_aspect_ratio(self) -> None:
        self.assertGreater(mouth_aspect_ratio(self.mouth_points), 0.0)

    def test_evaluate_mouth_open(self) -> None:
        self.assertTrue(evaluate_mouth_open(self.mouth_points, threshold=0.2))

    def test_lip_corner_spread_ratio(self) -> None:
        ratio = lip_corner_spread_ratio(self.landmarks)
        self.assertAlmostEqual(ratio, 0.5, places=6)

    def test_evaluate_smile(self) -> None:
        self.assertTrue(evaluate_smile(self.landmarks, threshold=0.4))


class TestHeadPoseAndIrisChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.pose_landmarks = [
            (0.0, 0.0),
            (4.0, 0.0),
            (2.0, 1.0),
            (1.0, 4.0),
            (3.0, 4.0),
            (2.0, 8.0),
        ]
        self.iris_points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]

    def test_pose_from_landmarks(self) -> None:
        pose = pose_from_landmarks(self.pose_landmarks)
        self.assertIsInstance(pose, HeadPose)
        self.assertGreater(pose.yaw, 0.0)

    def test_evaluate_head_pose(self) -> None:
        self.assertTrue(evaluate_head_pose(self.pose_landmarks, baseline=(0.0, 0.0, 0.0), threshold=5.0))

    def test_iris_centroid_and_tracking(self) -> None:
        centroid = iris_centroid(self.iris_points)
        self.assertTrue(np.allclose(centroid, np.array([2.5, 2.5], dtype=np.float32)))
        self.assertTrue(evaluate_iris_tracking(self.iris_points, target=(2.5, 2.5), tolerance=0.1))
        self.assertTrue(evaluate_iris_tracking(self.iris_points, target=IrisTarget(position=(2.5, 2.5), tolerance=0.1)))


class TestActiveLiveness(unittest.TestCase):
    def test_generate_session_token_is_deterministic_for_inputs(self) -> None:
        token_a = generate_session_token(timestamp="2026-04-19T00:00:00+00:00", nonce="abc")
        token_b = generate_session_token(timestamp="2026-04-19T00:00:00+00:00", nonce="abc")
        self.assertEqual(token_a, token_b)

    def test_orchestrator_runs_selected_challenges(self) -> None:
        orchestrator = ActiveLivenessOrchestrator(
            challenge_pool=["blink", "mouth"],
            min_challenges=2,
            max_challenges=2,
            rng=random.Random(0),
        )
        challenges = {
            "blink": [
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (4.0, 0.0),
                (2.0, -1.0),
                (1.0, -1.0),
            ],
            "mouth": [
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (3.0, 2.0),
                (4.0, 0.0),
                (3.0, -2.0),
                (2.0, -2.0),
                (1.0, -1.0),
            ],
        }
        session, results = orchestrator.create_and_run(
            challenges,
            baseline={"blink": 0.5, "mouth": 0.1},
            thresholds={"blink": 0.6, "mouth": 0.2},
        )
        self.assertEqual(len(session.challenge_names), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.passed for result in results))

    def test_orchestrator_rejects_bad_configuration(self) -> None:
        orchestrator = ActiveLivenessOrchestrator(challenge_pool=["blink"], min_challenges=2, max_challenges=2)
        with self.assertRaises(ActiveLivenessError):
            orchestrator.create_session()


if __name__ == "__main__":
    unittest.main()