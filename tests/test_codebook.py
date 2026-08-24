"""Tests for the Lloyd-Max codebook computation."""

import numpy as np
import pytest

from pyturboquant_cpu.codebook import (
    beta_pdf,
    get_codebook,
    lloyd_max_codebook,
)


class TestBetaPDF:
    """Tests for the Beta coordinate PDF."""

    def test_integrates_to_one(self):
        """PDF should integrate to 1 over [-1, 1]."""
        from scipy import integrate

        for dim in [8, 32, 128, 512]:
            result, _ = integrate.quad(lambda x: beta_pdf(x, dim), -1, 1)
            assert abs(result - 1.0) < 1e-8, (
                f"dim={dim}: integral={result}"
            )

    def test_symmetric(self):
        """PDF should be symmetric around 0."""
        x = np.linspace(0.01, 0.99, 50)
        for dim in [16, 64, 256]:
            pos = beta_pdf(x, dim)
            neg = beta_pdf(-x, dim)
            np.testing.assert_allclose(pos, neg, atol=1e-14)

    def test_zero_outside_support(self):
        """PDF should be zero outside [-1, 1]."""
        for dim in [16, 128]:
            assert beta_pdf(1.5, dim) == 0.0
            assert beta_pdf(-1.5, dim) == 0.0

    def test_positive_inside_support(self):
        """PDF should be positive inside (-1, 1)."""
        x = np.linspace(-0.99, 0.99, 100)
        for dim in [8, 64]:
            vals = beta_pdf(x, dim)
            assert np.all(vals > 0)


class TestLloydMaxCodebook:
    """Tests for the Lloyd-Max algorithm."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bits):
        """Should produce exactly 2^bits centroids."""
        centroids, boundaries = lloyd_max_codebook(128, bits, max_iter=100)
        assert centroids.shape == (2**bits,)
        assert boundaries.shape == (2**bits - 1,)

    @pytest.mark.parametrize("bits", [1, 2, 3])
    def test_centroids_symmetric(self, bits):
        """Centroids should be approximately symmetric around 0."""
        centroids, _ = lloyd_max_codebook(128, bits, max_iter=200)
        # Check c_i ≈ -c_{n-1-i}
        n = len(centroids)
        for i in range(n // 2):
            assert abs(centroids[i] + centroids[n - 1 - i]) < 1e-6

    def test_centroids_sorted(self):
        """Centroids should be in ascending order."""
        centroids, _ = lloyd_max_codebook(64, 3)
        assert np.all(np.diff(centroids) > 0)

    def test_boundaries_between_centroids(self):
        """Boundaries should be midpoints between consecutive centroids."""
        centroids, boundaries = lloyd_max_codebook(128, 2)
        expected = 0.5 * (centroids[:-1] + centroids[1:])
        np.testing.assert_allclose(boundaries, expected, atol=1e-12)

    def test_1bit_codebook_gaussian_limit(self):
        """For 1-bit and large d, centroids ≈ ±√(2/(πd))."""
        d = 1024
        centroids, _ = lloyd_max_codebook(d, 1, max_iter=300)
        expected = np.sqrt(2.0 / (np.pi * d))
        assert abs(centroids[0] + expected) < 0.01 * expected
        assert abs(centroids[1] - expected) < 0.01 * expected


class TestGetCodebook:
    """Tests for the cached codebook accessor."""

    def test_returns_same_object(self):
        """Calling get_codebook twice should return cached result."""
        c1, b1 = get_codebook(64, 2)
        c2, b2 = get_codebook(64, 2)
        assert c1 is c2
        assert b1 is b2

    def test_different_dims(self):
        """Different dimensions should produce different codebooks."""
        c32, _ = get_codebook(32, 2)
        c128, _ = get_codebook(128, 2)
        assert not np.allclose(c32, c128)
