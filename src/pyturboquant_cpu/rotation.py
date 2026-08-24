"""
Random orthogonal rotation matrices (Haar-distributed).

TurboQuant rotates input vectors with a uniformly random orthogonal matrix
so that each coordinate follows a known Beta distribution, enabling
data-oblivious scalar quantisation.
"""

from __future__ import annotations

import numpy as np


def generate_rotation_matrix(
    dim: int, seed: int | None = None
) -> np.ndarray:
    """Generate a uniformly random orthogonal matrix via QR decomposition.

    The matrix is drawn from the Haar measure on O(d) by computing the QR
    factorisation of a d×d matrix with i.i.d. N(0,1) entries, then
    correcting the signs so that the diagonal of R is positive (Stewart's
    algorithm).

    Parameters
    ----------
    dim : int
        Matrix dimension *d*.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    Q : ndarray, shape (d, d)
        Orthogonal matrix with det(Q) = ±1.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    # Correct signs to ensure uniform (Haar) distribution
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d[np.newaxis, :]
    return Q


def rotate(vectors: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply rotation: y = vectors @ Π^T.

    Parameters
    ----------
    vectors : ndarray, shape (..., d)
        Input vectors.
    rotation_matrix : ndarray, shape (d, d)
        Orthogonal rotation matrix Π.

    Returns
    -------
    ndarray, shape (..., d)
        Rotated vectors.
    """
    return vectors @ rotation_matrix.T


def inverse_rotate(
    vectors: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Apply inverse rotation: x = vectors @ Π.

    Since Π is orthogonal, Π^{-1} = Π^T, so inverse rotation is
    multiplication by Π (not transposed).

    Parameters
    ----------
    vectors : ndarray, shape (..., d)
        Rotated vectors.
    rotation_matrix : ndarray, shape (d, d)
        The same orthogonal matrix used for the forward rotation.

    Returns
    -------
    ndarray, shape (..., d)
        Vectors in the original basis.
    """
    return vectors @ rotation_matrix
