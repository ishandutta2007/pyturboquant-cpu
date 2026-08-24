"""
Core TurboQuant quantisation and dequantisation routines.

Provides two modes:

* **TurboQuant_MSE** — minimises mean-squared reconstruction error.
* **TurboQuant_Prod** — unbiased inner-product estimator (MSE stage +
  1-bit QJL residual correction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pyturboquant_cpu.codebook import get_codebook
from pyturboquant_cpu.packing import (
    pack_indices_fast,
    pack_signs,
    unpack_indices_fast,
    unpack_signs,
)
from pyturboquant_cpu.qjl import (
    generate_qjl_matrix,
    qjl_dequantize,
    qjl_quantize,
)
from pyturboquant_cpu.rotation import (
    generate_rotation_matrix,
    inverse_rotate,
    rotate,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QuantizedMSE:
    """Container for MSE-quantised vectors.

    Attributes
    ----------
    packed_indices : ndarray of uint8
        Bit-packed codebook indices for all vectors.
    norms : ndarray, shape (n,)
        Original L2 norms of each input vector.
    rotation_matrix : ndarray, shape (d, d)
        The random orthogonal matrix Π used for rotation.
    bits : int
        Bit-width per coordinate.
    dim : int
        Original vector dimension *d*.
    original_shape : tuple
        Shape of the input array (before flattening into 2-D).
    num_vectors : int
        Number of vectors that were quantised.
    centroids : ndarray, shape (2**bits,)
        Lloyd-Max codebook centroids.
    boundaries : ndarray, shape (2**bits - 1,)
        Decision boundaries for scalar quantisation.
    """

    packed_indices: np.ndarray
    norms: np.ndarray
    rotation_matrix: np.ndarray
    bits: int
    dim: int
    original_shape: tuple
    num_vectors: int
    centroids: np.ndarray
    boundaries: np.ndarray


@dataclass
class QuantizedProd:
    """Container for inner-product-optimal quantised vectors.

    Combines an MSE quantisation at ``(bits - 1)`` bits with a 1-bit QJL
    residual correction, giving an unbiased inner-product estimator at
    ``bits`` total bits per coordinate.

    Attributes
    ----------
    mse_component : QuantizedMSE
        MSE quantisation result using ``bits - 1`` bits.
    packed_signs : ndarray of uint8
        Packed QJL sign bits for each vector.
    residual_norms : ndarray, shape (n,)
        L2 norms of the residual vectors (γ).
    qjl_matrix : ndarray, shape (d, d)
        Random Gaussian matrix used for QJL.
    bits : int
        Total bit budget (the MSE component uses ``bits - 1``).
    """

    mse_component: QuantizedMSE
    packed_signs: np.ndarray
    residual_norms: np.ndarray
    qjl_matrix: np.ndarray
    bits: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_vectors(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, tuple, int, int]:
    """Reshape arbitrary-shaped input into (n, d) and return metadata."""
    vectors = np.asarray(vectors, dtype=np.float64)
    original_shape = vectors.shape
    if vectors.ndim == 1:
        vectors_2d = vectors[np.newaxis, :]
    else:
        vectors_2d = vectors.reshape(-1, vectors.shape[-1])
    n, d = vectors_2d.shape
    return vectors_2d, original_shape, n, d


def _scalar_quantize(
    values: np.ndarray,
    centroids: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Map each scalar value to the index of its nearest centroid.

    Uses ``np.searchsorted`` on boundaries for O(log k) lookup.

    Parameters
    ----------
    values : ndarray
        Scalar values to quantise.
    centroids : ndarray, shape (k,)
        Sorted codebook centroids.
    boundaries : ndarray, shape (k-1,)
        Decision boundaries.

    Returns
    -------
    indices : ndarray of uint32
        Codebook indices in ``[0, k)``.
    """
    indices = np.searchsorted(boundaries, values.ravel()).astype(np.uint32)
    return indices.reshape(values.shape)


def _scalar_dequantize(
    indices: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Map codebook indices back to centroid values."""
    return centroids[indices]


# ---------------------------------------------------------------------------
# MSE quantisation
# ---------------------------------------------------------------------------

def quantize_mse(
    vectors: np.ndarray,
    bits: int,
    dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> QuantizedMSE:
    """Quantise vectors using TurboQuant_MSE.

    Parameters
    ----------
    vectors : array_like, shape (..., d)
        Input vectors.  May be 1-D (single vector), 2-D (batch), or
        higher-dimensional.
    bits : int
        Bit-width per coordinate (1–8).
    dim : int or None, optional
        Override the detected vector dimension.  Rarely needed.
    seed : int or None, optional
        Random seed for the rotation matrix.

    Returns
    -------
    QuantizedMSE
        Packed quantisation result.

    Raises
    ------
    ValueError
        If *bits* is not in [1, 8].
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    vectors_2d, original_shape, n, d = _prepare_vectors(vectors)
    if dim is not None:
        d = dim

    # 1. Store norms and normalise to unit sphere
    norms = np.linalg.norm(vectors_2d, axis=1)
    # Avoid division by zero for zero vectors
    safe_norms = np.where(norms > 0, norms, 1.0)
    unit_vectors = vectors_2d / safe_norms[:, np.newaxis]

    # 2. Random rotation
    Pi = generate_rotation_matrix(d, seed=seed)
    rotated = rotate(unit_vectors, Pi)

    # 3. Scalar quantisation with precomputed codebook
    centroids, boundaries = get_codebook(d, bits)
    indices = _scalar_quantize(rotated, centroids, boundaries)

    # 4. Pack indices
    packed = pack_indices_fast(indices.ravel(), bits)

    return QuantizedMSE(
        packed_indices=packed,
        norms=norms,
        rotation_matrix=Pi,
        bits=bits,
        dim=d,
        original_shape=original_shape,
        num_vectors=n,
        centroids=centroids,
        boundaries=boundaries,
    )


def dequantize_mse(quantized: QuantizedMSE) -> np.ndarray:
    """Reconstruct vectors from a TurboQuant_MSE quantisation result.

    Parameters
    ----------
    quantized : QuantizedMSE
        Result of :func:`quantize_mse`.

    Returns
    -------
    ndarray
        Reconstructed vectors with the same shape as the original input.
    """
    n = quantized.num_vectors
    d = quantized.dim
    bits = quantized.bits

    # 1. Unpack indices
    total = n * d
    indices = unpack_indices_fast(quantized.packed_indices, bits, total)
    indices = indices.reshape(n, d)

    # 2. Map to centroids
    reconstructed_rotated = _scalar_dequantize(indices, quantized.centroids)

    # 3. Inverse rotation
    reconstructed_unit = inverse_rotate(
        reconstructed_rotated, quantized.rotation_matrix
    )

    # 4. Rescale by original norms
    reconstructed = reconstructed_unit * quantized.norms[:, np.newaxis]

    return reconstructed.reshape(quantized.original_shape)


# ---------------------------------------------------------------------------
# Inner-product-optimal quantisation
# ---------------------------------------------------------------------------

def quantize_prod(
    vectors: np.ndarray,
    bits: int,
    dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> QuantizedProd:
    """Quantise vectors using TurboQuant_Prod (unbiased inner products).

    Uses ``(bits - 1)`` bits for MSE quantisation plus 1 bit for QJL
    residual correction, for a total of ``bits`` bits per coordinate.

    Parameters
    ----------
    vectors : array_like, shape (..., d)
        Input vectors.
    bits : int
        Total bit-width per coordinate (2–8).  The MSE component uses
        ``bits - 1``.
    dim : int or None, optional
        Override detected dimension.
    seed : int or None, optional
        Random seed.  A deterministic derived seed is used for the QJL
        matrix so that the same top-level seed reproduces results.

    Returns
    -------
    QuantizedProd

    Raises
    ------
    ValueError
        If *bits* is not in [2, 8].
    """
    if not 2 <= bits <= 8:
        raise ValueError(
            f"bits must be in [2, 8] for Prod mode (need ≥1 bit for MSE "
            f"+ 1 bit for QJL), got {bits}"
        )

    vectors_2d, original_shape, n, d = _prepare_vectors(vectors)
    if dim is not None:
        d = dim

    # 1. Store norms and normalise
    norms = np.linalg.norm(vectors_2d, axis=1)
    safe_norms = np.where(norms > 0, norms, 1.0)
    unit_vectors = vectors_2d / safe_norms[:, np.newaxis]

    # 2. MSE quantisation at (bits - 1)
    mse_bits = bits - 1
    Pi = generate_rotation_matrix(d, seed=seed)
    centroids, boundaries = get_codebook(d, mse_bits)

    rotated = rotate(unit_vectors, Pi)
    indices = _scalar_quantize(rotated, centroids, boundaries)
    packed_mse = pack_indices_fast(indices.ravel(), mse_bits)

    mse_result = QuantizedMSE(
        packed_indices=packed_mse,
        norms=norms,
        rotation_matrix=Pi,
        bits=mse_bits,
        dim=d,
        original_shape=original_shape,
        num_vectors=n,
        centroids=centroids,
        boundaries=boundaries,
    )

    # 3. Compute residual on unit sphere
    reconstructed_rotated = _scalar_dequantize(indices, centroids)
    reconstructed_unit = inverse_rotate(reconstructed_rotated, Pi)
    residuals = unit_vectors - reconstructed_unit

    # 4. Residual norms (γ)
    residual_norms = np.linalg.norm(residuals, axis=1)

    # 5. Normalise residuals for QJL (avoid division by zero)
    safe_res_norms = np.where(residual_norms > 0, residual_norms, 1.0)
    normalised_residuals = residuals / safe_res_norms[:, np.newaxis]

    # 6. QJL quantisation of normalised residuals
    qjl_seed = (seed + 1_000_003) if seed is not None else None
    S = generate_qjl_matrix(d, seed=qjl_seed)
    signs = qjl_quantize(normalised_residuals, S)

    # 7. Pack sign bits
    packed_signs = np.concatenate(
        [pack_signs(signs[i]) for i in range(n)]
    )

    return QuantizedProd(
        mse_component=mse_result,
        packed_signs=packed_signs,
        residual_norms=residual_norms,
        qjl_matrix=S,
        bits=bits,
    )


def dequantize_prod(quantized: QuantizedProd) -> np.ndarray:
    """Reconstruct vectors from a TurboQuant_Prod quantisation result.

    The reconstruction is:  x̃ = ‖x‖ · (x̃_mse + γ · Q_qjl⁻¹(signs))

    Parameters
    ----------
    quantized : QuantizedProd
        Result of :func:`quantize_prod`.

    Returns
    -------
    ndarray
        Reconstructed vectors with the same shape as the original input.
    """
    mse = quantized.mse_component
    n = mse.num_vectors
    d = mse.dim

    # 1. MSE reconstruction (on unit sphere — set norms to 1)
    indices_total = n * d
    indices = unpack_indices_fast(
        mse.packed_indices, mse.bits, indices_total
    )
    indices = indices.reshape(n, d)
    reconstructed_rotated = _scalar_dequantize(indices, mse.centroids)
    mse_unit = inverse_rotate(reconstructed_rotated, mse.rotation_matrix)

    # 2. QJL dequantisation of residuals
    bytes_per_vector = (d + 7) // 8
    qjl_reconstruction = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        start = i * bytes_per_vector
        end = start + bytes_per_vector
        signs_i = unpack_signs(quantized.packed_signs[start:end], d)
        qjl_vec = qjl_dequantize(signs_i, quantized.qjl_matrix)
        qjl_reconstruction[i] = quantized.residual_norms[i] * qjl_vec

    # 3. Combine: unit reconstruction = MSE + QJL residual correction
    unit_reconstruction = mse_unit + qjl_reconstruction

    # 4. Rescale by original norms
    reconstructed = unit_reconstruction * mse.norms[:, np.newaxis]

    return reconstructed.reshape(mse.original_shape)
