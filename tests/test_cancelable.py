import unittest

import numpy as np

from security.cancelable import (
	CancelableTransformError,
	CancelableTransformer,
	apply_cancelable_transform,
	apply_cancelable_transform_batch,
	generate_projection_matrix,
)


class TestCancelableTransforms(unittest.TestCase):
	def test_projection_matrix_shape_and_orthonormality(self) -> None:
		matrix = generate_projection_matrix(embedding_dim=8, seed=42)
		self.assertEqual(matrix.shape, (8, 8))
		identity = matrix.T @ matrix
		self.assertTrue(np.allclose(identity, np.eye(8, dtype=np.float32), atol=1e-4))

	def test_transform_dimension_mismatch_raises(self) -> None:
		emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
		matrix = generate_projection_matrix(embedding_dim=4, seed=1)
		with self.assertRaises(CancelableTransformError):
			apply_cancelable_transform(emb, matrix)

	def test_transform_batch_returns_same_count(self) -> None:
		matrix = generate_projection_matrix(embedding_dim=4, seed=7)
		embeddings = [
			np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
			np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
		]
		transformed = apply_cancelable_transform_batch(embeddings, matrix)
		self.assertEqual(len(transformed), len(embeddings))
		self.assertTrue(all(vec.shape == (4,) for vec in transformed))


class TestCancelableTransformer(unittest.TestCase):
	def test_same_seed_produces_same_projection(self) -> None:
		t1 = CancelableTransformer(embedding_dim=8, seed=123)
		t2 = CancelableTransformer(embedding_dim=8, seed=123)
		self.assertTrue(np.allclose(t1.projection_matrix, t2.projection_matrix))

	def test_rotate_changes_projection(self) -> None:
		transformer = CancelableTransformer(embedding_dim=8, seed=100)
		original = transformer.projection_matrix
		rotated = transformer.rotate(seed=200)
		self.assertFalse(np.allclose(original, rotated))

	def test_transform_returns_1d_vector(self) -> None:
		transformer = CancelableTransformer(embedding_dim=4, seed=10)
		embedding = np.array([0.5, 0.2, 0.1, 0.9], dtype=np.float32)
		template = transformer.transform(embedding)
		self.assertEqual(template.shape, (4,))


if __name__ == "__main__":
	unittest.main()
