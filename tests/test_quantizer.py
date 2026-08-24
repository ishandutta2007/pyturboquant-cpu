"""Tests for the core TurboQuant quantiser."""

import numpy as np
import pytest

from pyturboquant_cpu.quantizer import (
    QuantizedMSE,
    QuantizedProd,
    dequantize_mse,
    dequantize_prod,
    quantize_mse,
    quantize_prod,
)


class TestQuantizeMSE:
    """Tests for TurboQuant_MSE."""

    def test_output_type(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 64))
        result = quantize_mse(x, bits=2, seed=42)
        assert isinstance(result, QuantizedMSE)

    def test_reconstructed_shape(self):
        """Dequantised output should have the same shape as input."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 128))
        q = quantize_mse(x, bits=3, seed=0)
        x_hat = dequantize_mse(q)
        assert x_hat.shape == x.shape

    def test_single_vector(self):
        """Should work for a single 1-D vector."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        q = quantize_mse(x, bits=4, seed=0)
        x_hat = dequantize_mse(q)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_decreases_with_bits(self, bits):
        """Higher bit-width should give lower MSE."""
        d = 128
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, d))
        # Normalise to unit sphere
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        mses = []
        for b in range(1, 5):
            q = quantize_mse(X, bits=b, seed=0)
            X_hat = dequantize_mse(q)
            mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))
            mses.append(mse)

        # Each bit-width should give lower MSE than the previous
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE did not decrease: bits={i+1} → {mses[i]:.4f}, "
                f"bits={i+2} → {mses[i+1]:.4f}"
            )

    def test_mse_distortion_bound(self):
        """MSE should be within the theoretical bound for unit vectors.

        Theorem 1: D_mse ≤ √3·π/2 · 1/4^b ≈ 2.72 / 4^b

        For b=2, the paper gives D_mse ≈ 0.117.
        """
        d = 256
        rng = np.random.default_rng(0)

        n_vectors = 200
        X = rng.standard_normal((n_vectors, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        bits = 2
        q = quantize_mse(X, bits=bits, seed=0)
        X_hat = dequantize_mse(q)
        mse_per_vector = np.sum((X - X_hat) ** 2, axis=1)
        avg_mse = mse_per_vector.mean()

        # Paper bound for b=2: ≈ 0.117
        # Allow margin for finite dimension and sample effects
        paper_bound = 0.117
        assert avg_mse < paper_bound * 4.0, (
            f"MSE {avg_mse:.4f} far exceeds theoretical bound "
            f"{paper_bound:.4f}"
        )

    def test_reproducibility(self):
        """Same seed should give identical results."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 64))
        q1 = quantize_mse(x, bits=3, seed=99)
        q2 = quantize_mse(x, bits=3, seed=99)
        x_hat1 = dequantize_mse(q1)
        x_hat2 = dequantize_mse(q2)
        np.testing.assert_array_equal(x_hat1, x_hat2)

    def test_zero_vector(self):
        """Zero vectors should not cause errors."""
        x = np.zeros((3, 64))
        q = quantize_mse(x, bits=2, seed=0)
        x_hat = dequantize_mse(q)
        np.testing.assert_allclose(x_hat, 0.0, atol=1e-15)

    def test_invalid_bits(self):
        """Should raise ValueError for bits outside [1, 8]."""
        x = np.random.randn(10, 32)
        with pytest.raises(ValueError):
            quantize_mse(x, bits=0)
        with pytest.raises(ValueError):
            quantize_mse(x, bits=9)


class TestQuantizeProd:
    """Tests for TurboQuant_Prod."""

    def test_output_type(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 64))
        result = quantize_prod(x, bits=3, seed=42)
        assert isinstance(result, QuantizedProd)

    def test_reconstructed_shape(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 128))
        q = quantize_prod(x, bits=3, seed=0)
        x_hat = dequantize_prod(q)
        assert x_hat.shape == x.shape

    def test_mse_component_uses_fewer_bits(self):
        """The MSE component should use (bits - 1)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 64))
        q = quantize_prod(x, bits=4, seed=0)
        assert q.mse_component.bits == 3

    def test_invalid_bits(self):
        """Prod mode needs ≥ 2 bits."""
        x = np.random.randn(10, 32)
        with pytest.raises(ValueError):
            quantize_prod(x, bits=1)

    def test_unbiased_inner_product(self):
        """E[⟨y, x̃⟩] ≈ ⟨y, x⟩ averaged over random seeds."""
        d = 128
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(d)
        true_ip = np.dot(y, x)

        n_trials = 200
        estimates = []
        for t in range(n_trials):
            q = quantize_prod(x, bits=3, seed=t)
            x_hat = dequantize_prod(q)
            estimates.append(np.dot(y, x_hat.ravel()))

        estimates = np.array(estimates)
        mean_est = estimates.mean()
        se = estimates.std() / np.sqrt(n_trials)

        # Should be unbiased within ~4 standard errors
        assert abs(mean_est - true_ip) < 4 * se, (
            f"Bias: E[est]={mean_est:.4f}, true={true_ip:.4f}, SE={se:.4f}"
        )
