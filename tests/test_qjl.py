"""Tests for the QJL (Quantized Johnson–Lindenstrauss) module."""

import math

import numpy as np
import pytest

from pyturboquant_cpu.qjl import (
    generate_qjl_matrix,
    qjl_dequantize,
    qjl_quantize,
)


class TestQJLMatrix:
    """Tests for QJL matrix generation."""

    def test_shape(self):
        S = generate_qjl_matrix(64, seed=0)
        assert S.shape == (64, 64)

    def test_reproducibility(self):
        S1 = generate_qjl_matrix(32, seed=42)
        S2 = generate_qjl_matrix(32, seed=42)
        np.testing.assert_array_equal(S1, S2)

    def test_approximate_gaussian(self):
        """Entries should be approximately N(0, 1)."""
        S = generate_qjl_matrix(256, seed=0)
        assert abs(S.mean()) < 0.1
        assert abs(S.std() - 1.0) < 0.1


class TestQJLUnbiasedness:
    """Test that QJL gives unbiased inner-product estimates."""

    def test_unbiased_inner_product(self):
        """E[⟨y, Q_qjl⁻¹(Q_qjl(x))⟩] ≈ ⟨y, x⟩ over many trials."""
        d = 128
        rng = np.random.default_rng(0)

        # Unit vector x
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        # Query vector y
        y = rng.standard_normal(d)

        true_ip = np.dot(y, x)

        n_trials = 2000
        estimates = np.zeros(n_trials)

        for t in range(n_trials):
            S = generate_qjl_matrix(d, seed=t)
            signs = qjl_quantize(x[np.newaxis, :], S)[0]
            x_hat = qjl_dequantize(signs, S)
            estimates[t] = np.dot(y, x_hat)

        mean_estimate = estimates.mean()
        # Should be within ~3 standard errors of the true value
        se = estimates.std() / math.sqrt(n_trials)
        assert abs(mean_estimate - true_ip) < 4 * se, (
            f"Bias detected: E[est]={mean_estimate:.4f}, "
            f"true={true_ip:.4f}, SE={se:.4f}"
        )

    def test_variance_bound(self):
        """Var(⟨y, Q_qjl⁻¹(Q_qjl(x))⟩) ≤ (π/2d) · ‖y‖²."""
        d = 128
        rng = np.random.default_rng(1)

        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(d)

        n_trials = 2000
        estimates = np.zeros(n_trials)

        for t in range(n_trials):
            S = generate_qjl_matrix(d, seed=t + 10000)
            signs = qjl_quantize(x[np.newaxis, :], S)[0]
            x_hat = qjl_dequantize(signs, S)
            estimates[t] = np.dot(y, x_hat)

        var_estimate = estimates.var()
        var_bound = (math.pi / (2 * d)) * np.dot(y, y)

        # Allow some slack for finite samples
        assert var_estimate < var_bound * 1.5, (
            f"Variance {var_estimate:.4f} exceeds bound {var_bound:.4f}"
        )
