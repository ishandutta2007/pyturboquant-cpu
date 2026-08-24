"""
Quantized Johnson–Lindenstrauss (QJL) 1-bit inner-product quantiser.

QJL provides an *unbiased* estimator for ⟨y, x⟩ using only 1 bit per
coordinate.  It is used as the second stage of TurboQuant_Prod to
correct the inner-product bias introduced by the MSE-optimal quantiser.

Reference
---------
Zandieh et al., Definition 1 and Lemma 4 of arXiv:2504.19874.
"""

from __future__ import annotations

import math

import numpy as np


def generate_qjl_matrix(
    dim: int, seed: int | None = None
) -> np.ndarray:
    """Generate a d×d random Gaussian matrix for QJL.

    Each entry is drawn i.i.d. from N(0, 1).

    Parameters
    ----------
    dim : int
        Matrix dimension *d*.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray, shape (d, d)
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dim, dim))


def qjl_quantize(
    vectors: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Apply QJL quantisation: sign(S · x).

    Parameters
    ----------
    vectors : ndarray, shape (..., d)
        Input vectors (should already be normalised / residuals).
    S : ndarray, shape (d, d)
        Random Gaussian matrix.

    Returns
    -------
    signs : ndarray, shape (..., d)
        Array of -1.0 / +1.0 values.
    """
    # vectors @ S.T gives (..., d)
    projected = vectors @ S.T
    signs = np.sign(projected)
    # Replace exact zeros (extremely unlikely) with +1
    signs[signs == 0] = 1.0
    return signs


def qjl_dequantize(
    signs: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """QJL inverse map: Q_qjl^{-1}(z) = √(π/2) · S^T · z / d.

    Parameters
    ----------
    signs : ndarray, shape (..., d)
        Array of -1 / +1 values.
    S : ndarray, shape (d, d)
        The same random matrix used during quantisation.

    Returns
    -------
    reconstructed : ndarray, shape (..., d)
    """
    d = S.shape[0]
    scale = math.sqrt(math.pi / 2.0) / d
    # signs @ S gives (..., d)
    return scale * (signs @ S)
