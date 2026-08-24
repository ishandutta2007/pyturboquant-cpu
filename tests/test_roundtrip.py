"""End-to-end roundtrip tests for TurboQuant."""

import numpy as np
import pytest

from pyturboquant_cpu import (
    dequantize_mse,
    dequantize_prod,
    quantize_mse,
    quantize_prod,
)


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_mse_roundtrip_quality(self):
        """MSE reconstruction should be close to original for high bit-width."""
        d = 128
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, d))

        q = quantize_mse(X, bits=4, seed=0)
        X_hat = dequantize_mse(q)

        # Relative error per vector
        norms = np.linalg.norm(X, axis=1)
        errors = np.linalg.norm(X - X_hat, axis=1)
        relative_errors = errors / np.maximum(norms, 1e-10)

        # At 4 bits the relative error for random Gaussian vectors is moderate
        assert np.mean(relative_errors) < 0.55

    def test_prod_roundtrip_quality(self):
        """Prod mode should also reconstruct reasonably well."""
        d = 128
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, d))

        q = quantize_prod(X, bits=4, seed=0)
        X_hat = dequantize_prod(q)

        norms = np.linalg.norm(X, axis=1)
        errors = np.linalg.norm(X - X_hat, axis=1)
        relative_errors = errors / np.maximum(norms, 1e-10)

        assert np.mean(relative_errors) < 0.85

    def test_3d_input_shape(self):
        """Should handle 3-D inputs (e.g., (layers, tokens, dim))."""
        shape = (4, 8, 64)
        rng = np.random.default_rng(0)
        X = rng.standard_normal(shape)

        q = quantize_mse(X, bits=3, seed=0)
        X_hat = dequantize_mse(q)
        assert X_hat.shape == shape

    def test_compression_ratio(self):
        """Check that packed indices are smaller than the original."""
        d = 128
        n = 100
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d))

        q = quantize_mse(X, bits=2, seed=0)

        original_bytes = X.nbytes  # n * d * 8 (float64)
        packed_bytes = q.packed_indices.nbytes + q.norms.nbytes

        # 2 bits per coordinate vs 64 bits = 32x theoretical max
        # We also store norms, but still should be much smaller
        assert packed_bytes < original_bytes / 10

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_theoretical_values(self, bits):
        """Compare average MSE against paper's theoretical values.

        Paper (Theorem 1) for unit vectors:
            b=1: D_mse ≈ 0.36
            b=2: D_mse ≈ 0.117
            b=3: D_mse ≈ 0.03
            b=4: D_mse ≈ 0.009
        """
        theoretical = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

        d = 256
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        q = quantize_mse(X, bits=bits, seed=0)
        X_hat = dequantize_mse(q)

        mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))

        # Should be within 2x of theoretical bound (generous for finite d)
        assert mse < theoretical[bits] * 2.5, (
            f"bits={bits}: MSE={mse:.4f}, "
            f"expected ≤ {theoretical[bits] * 2.5:.4f}"
        )

    def test_import_api(self):
        """Public API should be importable from package root."""
        from pyturboquant_cpu import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
