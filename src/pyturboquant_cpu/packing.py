"""
Bit-packing and unpacking utilities.

Quantised indices (b bits each) and QJL sign bits are packed into
contiguous ``uint8`` arrays for compact storage.
"""

from __future__ import annotations

import numpy as np


def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack an array of integer indices into a compact ``uint8`` byte array.

    Each index value must be in ``[0, 2**bits)``.  The indices are stored
    in a flat bit-stream, packed from the LSB of each output byte.

    Parameters
    ----------
    indices : ndarray of int
        Flat array of index values.
    bits : int
        Number of bits per index (1–8).

    Returns
    -------
    packed : ndarray of uint8
        Byte array containing the packed bits.
    """
    indices = np.asarray(indices, dtype=np.uint32).ravel()
    n = indices.size
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8

    packed = np.zeros(total_bytes, dtype=np.uint8)

    # Build a flat bit stream
    for b in range(bits):
        # Extract bit `b` from each index
        bit_values = ((indices >> b) & 1).astype(np.uint8)
        for i, bv in enumerate(bit_values):
            flat_bit_pos = i * bits + b
            byte_idx = flat_bit_pos // 8
            bit_idx = flat_bit_pos % 8
            packed[byte_idx] |= bv << bit_idx

    return packed


def unpack_indices(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpack integer indices from a packed ``uint8`` byte array.

    Parameters
    ----------
    packed : ndarray of uint8
        Byte array produced by :func:`pack_indices`.
    bits : int
        Number of bits per index (1–8).
    count : int
        Number of indices to unpack.

    Returns
    -------
    indices : ndarray of uint32
        Recovered index values.
    """
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    indices = np.zeros(count, dtype=np.uint32)

    for i in range(count):
        val = np.uint32(0)
        for b in range(bits):
            flat_bit_pos = i * bits + b
            byte_idx = flat_bit_pos // 8
            bit_idx = flat_bit_pos % 8
            bit_val = (packed[byte_idx] >> bit_idx) & 1
            val |= np.uint32(bit_val) << np.uint32(b)
        indices[i] = val

    return indices


def pack_indices_fast(indices: np.ndarray, bits: int) -> np.ndarray:
    """Vectorised bit-packing for common bit-widths.

    Falls back to :func:`pack_indices` for uncommon widths.

    Parameters
    ----------
    indices : ndarray of int
        Flat array of index values.
    bits : int
        Number of bits per index.

    Returns
    -------
    packed : ndarray of uint8
    """
    indices = np.asarray(indices, dtype=np.uint8).ravel()

    if bits == 8:
        return indices.copy()

    if bits == 4:
        # Pack two 4-bit values per byte
        n = indices.size
        padded = indices
        if n % 2 != 0:
            padded = np.append(indices, np.uint8(0))
        low = padded[0::2] & 0x0F
        high = (padded[1::2] & 0x0F) << 4
        return (low | high).astype(np.uint8)

    if bits == 2:
        # Pack four 2-bit values per byte
        n = indices.size
        pad_len = (4 - n % 4) % 4
        if pad_len:
            padded = np.append(indices, np.zeros(pad_len, dtype=np.uint8))
        else:
            padded = indices
        a = padded[0::4] & 0x03
        b = (padded[1::4] & 0x03) << 2
        c = (padded[2::4] & 0x03) << 4
        d = (padded[3::4] & 0x03) << 6
        return (a | b | c | d).astype(np.uint8)

    if bits == 1:
        return pack_signs_raw(indices)

    # General fallback
    return pack_indices(indices.astype(np.uint32), bits)


def unpack_indices_fast(
    packed: np.ndarray, bits: int, count: int
) -> np.ndarray:
    """Vectorised unpacking for common bit-widths.

    Parameters
    ----------
    packed : ndarray of uint8
    bits : int
    count : int

    Returns
    -------
    indices : ndarray of uint32
    """
    packed = np.asarray(packed, dtype=np.uint8).ravel()

    if bits == 8:
        return packed[:count].astype(np.uint32)

    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        interleaved = np.empty(packed.size * 2, dtype=np.uint32)
        interleaved[0::2] = low
        interleaved[1::2] = high
        return interleaved[:count]

    if bits == 2:
        a = packed & 0x03
        b = (packed >> 2) & 0x03
        c = (packed >> 4) & 0x03
        d = (packed >> 6) & 0x03
        interleaved = np.empty(packed.size * 4, dtype=np.uint32)
        interleaved[0::4] = a
        interleaved[1::4] = b
        interleaved[2::4] = c
        interleaved[3::4] = d
        return interleaved[:count]

    if bits == 1:
        signs_raw = unpack_signs_raw(packed, count)
        return signs_raw.astype(np.uint32)

    return unpack_indices(packed, bits, count)


def pack_signs_raw(bits_array: np.ndarray) -> np.ndarray:
    """Pack a {0,1} bit array into bytes.

    Parameters
    ----------
    bits_array : ndarray
        Array of 0/1 values.

    Returns
    -------
    packed : ndarray of uint8
    """
    bits_array = np.asarray(bits_array, dtype=np.uint8).ravel()
    n = bits_array.size
    pad_len = (8 - n % 8) % 8
    if pad_len:
        bits_array = np.append(
            bits_array, np.zeros(pad_len, dtype=np.uint8)
        )
    reshaped = bits_array.reshape(-1, 8)
    multipliers = (1 << np.arange(8, dtype=np.uint8))[np.newaxis, :]
    return (reshaped * multipliers).sum(axis=1).astype(np.uint8)


def unpack_signs_raw(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack bytes into a {0,1} bit array.

    Parameters
    ----------
    packed : ndarray of uint8
    count : int

    Returns
    -------
    bits_array : ndarray of uint8
        Array of 0/1 values.
    """
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    bits = np.unpackbits(packed, bitorder="little")
    return bits[:count]


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """Pack a {-1, +1} sign array into bytes.

    +1 is stored as bit 1, -1 as bit 0.

    Parameters
    ----------
    signs : ndarray
        Array of -1 / +1 values.

    Returns
    -------
    packed : ndarray of uint8
    """
    bits = ((np.asarray(signs).ravel() + 1) // 2).astype(np.uint8)
    return pack_signs_raw(bits)


def unpack_signs(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack bytes into a {-1, +1} sign array.

    Parameters
    ----------
    packed : ndarray of uint8
    count : int
        Number of sign values to recover.

    Returns
    -------
    signs : ndarray of float64
        Array of -1.0 / +1.0 values.
    """
    bits = unpack_signs_raw(packed, count)
    return bits.astype(np.float64) * 2.0 - 1.0
