"""Tests for bit-packing utilities."""

import numpy as np
import pytest

from pyturboquant_cpu.packing import (
    pack_indices,
    pack_indices_fast,
    pack_signs,
    unpack_indices,
    unpack_indices_fast,
    unpack_signs,
)


class TestPackUnpackIndices:
    """Tests for the general pack/unpack routines."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_roundtrip(self, bits):
        """Pack then unpack should recover original indices."""
        rng = np.random.default_rng(42)
        n = 100
        max_val = (1 << bits)
        indices = rng.integers(0, max_val, size=n, dtype=np.uint32)
        packed = pack_indices(indices, bits)
        recovered = unpack_indices(packed, bits, n)
        np.testing.assert_array_equal(recovered, indices)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_roundtrip_fast(self, bits):
        """Fast pack/unpack should match general version."""
        rng = np.random.default_rng(42)
        n = 200
        max_val = (1 << bits)
        indices = rng.integers(0, max_val, size=n, dtype=np.uint32)
        packed = pack_indices_fast(indices, bits)
        recovered = unpack_indices_fast(packed, bits, n)
        np.testing.assert_array_equal(recovered, indices)


class TestPackUnpackSigns:
    """Tests for sign bit packing."""

    def test_roundtrip(self):
        """Pack then unpack should recover original signs."""
        rng = np.random.default_rng(0)
        n = 256
        signs = rng.choice([-1.0, 1.0], size=n)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, n)
        np.testing.assert_array_equal(recovered, signs)

    def test_roundtrip_odd_count(self):
        """Should work when count is not a multiple of 8."""
        signs = np.array([1, -1, 1, 1, -1], dtype=np.float64)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 5)
        np.testing.assert_array_equal(recovered, signs)

    def test_all_positive(self):
        """All +1 signs."""
        signs = np.ones(16)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 16)
        np.testing.assert_array_equal(recovered, signs)

    def test_all_negative(self):
        """All -1 signs."""
        signs = -np.ones(16)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 16)
        np.testing.assert_array_equal(recovered, signs)
