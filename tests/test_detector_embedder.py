import unittest

import numpy as np

from core.detector import DetectionError, FaceDetector, crop_face
from core.embedder import EmbeddingError, FaceEmbedder, normalize_embedding


class FakeFace:
    def __init__(self, bbox, det_score=0.9, landmark=None, embedding=None):
        self.bbox = bbox
        self.det_score = det_score
        self.landmark = landmark
        self.embedding = embedding


class FakeDetectionModel:
    def get(self, frame):
        return [FakeFace(bbox=(1, 1, 3, 3), det_score=0.95)]


class FakeEmbeddingModel:
    def get(self, face_image):
        return [FakeFace(bbox=(0, 0, 0, 0), embedding=np.array([3.0, 4.0], dtype=np.float32))]


class TestDetectorUtilities(unittest.TestCase):
    def test_crop_face_uses_absolute_bbox(self) -> None:
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        cropped = crop_face(image, (1, 1, 4, 4))
        self.assertEqual(cropped.shape, (3, 3))

    def test_crop_face_rejects_invalid_bbox(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(DetectionError):
            crop_face(image, (1, 2, 3))

    def test_detect_face_with_injected_model(self) -> None:
        detector = FaceDetector(model=FakeDetectionModel(), margin=0.0)
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        result = detector.detect(image)
        self.assertEqual(result.bbox, (1, 1, 3, 3))
        self.assertEqual(result.face.shape[:2], (2, 2))


class TestEmbedderUtilities(unittest.TestCase):
    def test_normalize_embedding(self) -> None:
        vector = normalize_embedding([3.0, 4.0])
        self.assertTrue(np.allclose(vector, np.array([0.6, 0.8], dtype=np.float32)))

    def test_normalize_embedding_rejects_zero_vector(self) -> None:
        with self.assertRaises(EmbeddingError):
            normalize_embedding([0.0, 0.0])

    def test_extract_embedding_with_1d_input_short_circuits(self) -> None:
        embedder = FaceEmbedder(model=FakeEmbeddingModel())
        result = embedder.extract_embedding(np.array([3.0, 4.0], dtype=np.float32))
        self.assertTrue(np.allclose(result, np.array([0.6, 0.8], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()