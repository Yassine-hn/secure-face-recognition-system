import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.enrollment import EnrollmentError, EnrollmentService, average_embeddings, collect_embeddings
from security.cancelable import CancelableTransformer
from security.database import DatabaseError, EnrollmentDatabase
from security.key_manager import KeyManagerError, get_key_from_env, normalize_db_key


class FakeDetector:
    def detect_face(self, frame):
        return frame.get("face")


class FakeDetectionResult:
    def __init__(self, embedding):
        self.embedding = embedding
        self.face = np.array([9.0, 9.0, 9.0], dtype=np.float32)


class FakeEmbeddingDetector:
    def detect(self, frame):
        return FakeDetectionResult(frame.get("embedding"))


class FakeEmbedder:
    def extract_embedding(self, face):
        return face


class TestKeyManagerHelpers(unittest.TestCase):
    def test_normalize_db_key_rejects_empty(self) -> None:
        with self.assertRaises(KeyManagerError):
            normalize_db_key("   ")

    def test_get_key_from_env_reads_value(self) -> None:
        os.environ["DB_KEY"] = "env-secret"
        try:
            self.assertEqual(get_key_from_env(), "env-secret")
        finally:
            os.environ.pop("DB_KEY", None)

    def test_get_db_key_uses_env_when_available(self) -> None:
        from security.key_manager import get_db_key
        os.environ["DB_KEY"] = "custom-test-secret"
        try:
            self.assertEqual(
                get_db_key(service_name="test_service_unused", username="test_user_unused"),
                "custom-test-secret"
            )
        finally:
            os.environ.pop("DB_KEY", None)

    def test_get_db_key_uses_demo_fallback_in_dev_mode(self) -> None:
        from security.key_manager import get_db_key
        old_key = os.environ.pop("DB_KEY", None)
        old_mode = os.environ.get("SFRS_MODE")
        os.environ["SFRS_MODE"] = "development"
        try:
            key = get_db_key(service_name="nonexistent_service_12345", username="user_none")
            self.assertTrue(len(key) > 0)
        finally:
            if old_key is not None:
                os.environ["DB_KEY"] = old_key
            if old_mode is not None:
                os.environ["SFRS_MODE"] = old_mode
            else:
                os.environ.pop("SFRS_MODE", None)


class TestEnrollmentDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "embeddings.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upsert_and_fetch_roundtrip(self) -> None:
        with EnrollmentDatabase(
            db_path=self.db_path,
            db_key="test-key",
            allow_plaintext_fallback=True,
        ) as database:
            record = database.upsert_enrollment(
                "student-1",
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                metadata={"group": "A"},
            )
            self.assertEqual(record.student_id, "student-1")
            self.assertEqual(record.metadata["group"], "A")
            self.assertTrue(np.allclose(record.template, np.array([1.0, 2.0, 3.0], dtype=np.float32)))

            fetched = database.get_enrollment("student-1")
            self.assertEqual(fetched.student_id, "student-1")
            self.assertEqual(len(database.list_enrollments()), 1)

    def test_upsert_rejects_empty_student_id(self) -> None:
        with EnrollmentDatabase(
            db_path=self.db_path,
            db_key="test-key",
            allow_plaintext_fallback=True,
        ) as database:
            with self.assertRaises(DatabaseError):
                database.upsert_enrollment("   ", np.array([1.0, 2.0], dtype=np.float32))


class TestEnrollmentService(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "embeddings.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_average_embeddings(self) -> None:
        result = average_embeddings([
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([3.0, 2.0], dtype=np.float32),
        ])
        self.assertTrue(np.allclose(result, np.array([2.0, 1.0], dtype=np.float32)))

    def test_collect_embeddings_and_enroll_from_capture(self) -> None:
        frames = [
            {"face": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
            {"face": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
            {"face": np.array([0.0, 0.0, 1.0], dtype=np.float32)},
        ]

        def frame_source():
            return frames.pop(0) if frames else None

        with EnrollmentDatabase(
            db_path=self.db_path,
            db_key="test-key",
            allow_plaintext_fallback=True,
        ) as database:
            transformer = CancelableTransformer(embedding_dim=3, seed=11)
            service = EnrollmentService(database=database, transformer=transformer)
            collected = collect_embeddings(
                frame_source=frame_source,
                face_detector=FakeDetector(),
                embedder=FakeEmbedder(),
                sample_count=3,
                max_attempts=5,
            )
            self.assertEqual(len(collected), 3)

            record = service.enroll_from_capture(
                "student-2",
                frame_source=lambda: {"face": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
                face_detector=FakeDetector(),
                embedder=FakeEmbedder(),
                sample_count=1,
                max_attempts=1,
                metadata={"source": "camera"},
            )
            self.assertEqual(record.student_id, "student-2")
            self.assertEqual(record.metadata["source"], "camera")
            self.assertEqual(record.template.shape, (3,))

    def test_collect_embeddings_uses_detector_embedding_when_available(self) -> None:
        frames = [{"embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32)}]

        def frame_source():
            return frames.pop(0) if frames else None

        embeddings = collect_embeddings(
            frame_source=frame_source,
            face_detector=FakeEmbeddingDetector(),
            embedder=FakeEmbedder(),
            sample_count=1,
            max_attempts=1,
        )
        self.assertTrue(np.allclose(embeddings[0], np.array([1.0, 2.0, 3.0], dtype=np.float32)))

    def test_collect_embeddings_raises_when_not_enough_samples(self) -> None:
        with self.assertRaises(EnrollmentError):
            collect_embeddings(
                frame_source=lambda: None,
                face_detector=FakeDetector(),
                embedder=FakeEmbedder(),
                sample_count=1,
                max_attempts=1,
            )

    def test_enroll_rejects_empty_student_id(self) -> None:
        with EnrollmentDatabase(
            db_path=self.db_path,
            db_key="test-key",
            allow_plaintext_fallback=True,
        ) as database:
            service = EnrollmentService(database=database)
            with self.assertRaises(EnrollmentError):
                service.enroll_embedding("  ", np.array([1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()