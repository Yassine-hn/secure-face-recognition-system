import unittest

import numpy as np

from core.matcher import FaceMatcher
from core.recognition import RecognitionService


class FakeDetector:
    def detect_face(self, frame):
        return frame["face"]


class FakeEmbedder:
    def extract_embedding(self, face):
        return face


class TestRecognitionService(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = FaceMatcher(threshold=0.8)
        self.service = RecognitionService(
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            matcher=self.matcher,
        )
        self.enrolled = {
            "student-1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "student-2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def test_recognize_accepts_matching_template(self) -> None:
        frame = {"face": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        result = self.service.recognize(frame, self.enrolled)
        self.assertTrue(result.accepted)
        self.assertEqual(result.student_id, "student-1")
        self.assertAlmostEqual(result.score, 1.0, places=6)

    def test_recognize_rejects_non_matching_template(self) -> None:
        frame = {"face": np.array([0.0, 0.0, 1.0], dtype=np.float32)}
        result = self.service.recognize(frame, self.enrolled)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.student_id)
        self.assertIsNotNone(result.match)


if __name__ == "__main__":
    unittest.main()