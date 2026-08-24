"""Tests for random rotation matrix generation."""

import numpy as np
import pytest

from pyturboquant_cpu.rotation import (
    generate_rotation_matrix,
    inverse_rotate,
    rotate,
)


class TestGenerateRotationMatrix:
    """Tests for generate_rotation_matrix."""

    @pytest.mark.parametrize("dim", [4, 16, 64])
    def test_orthogonality(self, dim):
        """Q^T Q should equal the identity matrix."""
        Q = generate_rotation_matrix(dim, seed=42)
        identity = Q.T @ Q
        np.testing.assert_allclose(identity, np.eye(dim), atol=1e-12)

    @pytest.mark.parametrize("dim", [4, 16, 64])
    def test_determinant(self, dim):
        """Determinant should be ±1."""
        Q = generate_rotation_matrix(dim, seed=42)
        det = np.linalg.det(Q)
        assert abs(abs(det) - 1.0) < 1e-10

    def test_reproducibility(self):
        """Same seed should produce the same matrix."""
        Q1 = generate_rotation_matrix(32, seed=123)
        Q2 = generate_rotation_matrix(32, seed=123)
        np.testing.assert_array_equal(Q1, Q2)

    def test_different_seeds(self):
        """Different seeds should produce different matrices."""
        Q1 = generate_rotation_matrix(32, seed=1)
        Q2 = generate_rotation_matrix(32, seed=2)
        assert not np.allclose(Q1, Q2)


class TestRotateInverseRotate:
    """Tests for rotate and inverse_rotate."""

    def test_roundtrip_single_vector(self):
        """Rotating and inverse-rotating should recover the original."""
        d = 64
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d)
        Q = generate_rotation_matrix(d, seed=99)
        y = rotate(x, Q)
        x_recovered = inverse_rotate(y, Q)
        np.testing.assert_allclose(x_recovered, x, atol=1e-12)

    def test_roundtrip_batch(self):
        """Should work for a batch of vectors."""
        n, d = 50, 128
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d))
        Q = generate_rotation_matrix(d, seed=7)
        Y = rotate(X, Q)
        X_recovered = inverse_rotate(Y, Q)
        np.testing.assert_allclose(X_recovered, X, atol=1e-11)

    def test_norm_preservation(self):
        """Rotation should preserve vector norms."""
        d = 64
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, d))
        Q = generate_rotation_matrix(d, seed=5)
        Y = rotate(X, Q)
        original_norms = np.linalg.norm(X, axis=1)
        rotated_norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-12)
