import unittest

import numpy as np

from core.matcher import FaceMatcher, MatcherError, batch_scores, cosine_similarity


class TestCosineSimilarity(unittest.TestCase):
	def test_identical_vectors_score_one(self) -> None:
		a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
		b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
		self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=6)

	def test_orthogonal_vectors_score_zero(self) -> None:
		a = np.array([1.0, 0.0], dtype=np.float32)
		b = np.array([0.0, 1.0], dtype=np.float32)
		self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

	def test_shape_mismatch_raises(self) -> None:
		with self.assertRaises(MatcherError):
			cosine_similarity([1.0, 2.0], [1.0])

	def test_zero_norm_vector_raises(self) -> None:
		with self.assertRaises(MatcherError):
			cosine_similarity([0.0, 0.0], [1.0, 1.0])


class TestFaceMatcher(unittest.TestCase):
	def setUp(self) -> None:
		self.query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
		self.enrolled = {
			"student_a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
			"student_b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
		}

	def test_find_best_match_returns_top_candidate(self) -> None:
		matcher = FaceMatcher(threshold=0.5)
		result = matcher.find_best_match(self.query, self.enrolled)
		self.assertIsNotNone(result)
		assert result is not None
		self.assertEqual(result.student_id, "student_a")
		self.assertTrue(result.is_match)
		self.assertAlmostEqual(result.score, 1.0, places=6)

	def test_identify_returns_none_when_below_threshold(self) -> None:
		matcher = FaceMatcher(threshold=0.99)
		student_id, score = matcher.identify(
			np.array([0.0, 1.0, 0.0], dtype=np.float32),
			{"student_a": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
		)
		self.assertIsNone(student_id)
		self.assertAlmostEqual(score, 0.0, places=6)

	def test_set_threshold_rejects_out_of_range_values(self) -> None:
		matcher = FaceMatcher(threshold=0.5)
		with self.assertRaises(MatcherError):
			matcher.set_threshold(1.01)

	def test_find_best_match_returns_none_when_no_templates(self) -> None:
		matcher = FaceMatcher(threshold=0.5)
		self.assertIsNone(matcher.find_best_match(self.query, {}))

	def test_batch_scores_computes_all_identities(self) -> None:
		scores = batch_scores(self.query, self.enrolled)
		self.assertEqual(set(scores.keys()), {"student_a", "student_b"})
		self.assertGreater(scores["student_a"], scores["student_b"])


if __name__ == "__main__":
	unittest.main()
