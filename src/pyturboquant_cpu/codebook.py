"""
Lloyd-Max codebook computation for the Beta coordinate distribution.

After a random orthogonal rotation, each coordinate of a unit-norm vector
in R^d follows a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1].

This module solves the 1-D Lloyd-Max (k-means) quantisation problem for
that distribution and caches the results for reuse.
"""

from __future__ import annotations

import functools
import math
from typing import Tuple

import numpy as np
from scipy import integrate, special


def beta_pdf(x: np.ndarray | float, dim: int) -> np.ndarray | float:
    """Probability density of a single coordinate after random rotation.

    For a unit vector uniformly distributed on S^{d-1}, each coordinate
    follows the distribution:

        f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

    for x ∈ [-1, 1].

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    dim : int
        Ambient dimension *d*.

    Returns
    -------
    array_like
        PDF values.
    """
    x = np.asarray(x, dtype=np.float64)
    d = dim

    # Use log-gamma for numerical stability
    log_norm = (
        special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - special.gammaln((d - 1) / 2.0)
    )

    exponent = (d - 3) / 2.0

    # Mask values outside support
    mask = np.abs(x) < 1.0
    result = np.zeros_like(x, dtype=np.float64)
    if exponent == 0:
        # d = 3: uniform on [-1, 1]
        result[mask] = math.exp(log_norm)
    else:
        safe_x = np.where(mask, x, 0.0)
        log_body = exponent * np.log1p(-safe_x**2)
        result[mask] = np.exp(log_norm + log_body[mask])

    return result


def _conditional_expectation(a: float, b: float, dim: int) -> float:
    """Compute E[X | a ≤ X ≤ b] for the Beta coordinate distribution.

    Parameters
    ----------
    a, b : float
        Interval endpoints within [-1, 1].
    dim : int
        Ambient dimension *d*.

    Returns
    -------
    float
        Conditional mean.
    """
    if b - a < 1e-15:
        return (a + b) / 2.0

    # Numerator: ∫_a^b x · f(x) dx
    numerator, _ = integrate.quad(lambda x: x * beta_pdf(x, dim), a, b)
    # Denominator: ∫_a^b f(x) dx
    denominator, _ = integrate.quad(lambda x: beta_pdf(x, dim), a, b)

    if abs(denominator) < 1e-30:
        return (a + b) / 2.0

    return numerator / denominator


def lloyd_max_codebook(
    dim: int,
    bits: int,
    max_iter: int = 300,
    tol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal codebook for the Beta coordinate distribution.

    Solves the 1-D k-means problem for the distribution induced by random
    rotation on the unit sphere in dimension *dim*, using *2^bits* centroids.

    Parameters
    ----------
    dim : int
        Vector dimension *d*.
    bits : int
        Bit-width per coordinate.  Produces ``2**bits`` centroids.
    max_iter : int, optional
        Maximum Lloyd iterations.
    tol : float, optional
        Convergence tolerance on centroid movement.

    Returns
    -------
    centroids : ndarray, shape (2**bits,)
        Sorted codebook centroids.
    boundaries : ndarray, shape (2**bits - 1,)
        Decision boundaries (midpoints between consecutive centroids).
    """
    n_levels = 1 << bits  # 2^bits

    # Initialise centroids uniformly in [-1, 1]
    centroids = np.linspace(-1.0, 1.0, n_levels + 2)[1:-1].copy()

    for _ in range(max_iter):
        # Boundaries = midpoints between consecutive centroids
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

        # Build full interval edges: [-1, b_1, b_2, ..., b_{n-1}, 1]
        edges = np.empty(n_levels + 1, dtype=np.float64)
        edges[0] = -1.0
        edges[-1] = 1.0
        edges[1:-1] = boundaries

        # Update each centroid to the conditional expectation over its cell
        new_centroids = np.empty_like(centroids)
        for i in range(n_levels):
            new_centroids[i] = _conditional_expectation(
                edges[i], edges[i + 1], dim
            )

        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if delta < tol:
            break

    # Final boundaries
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.astype(np.float64), boundaries.astype(np.float64)


@functools.lru_cache(maxsize=256)
def get_codebook(
    dim: int, bits: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a cached Lloyd-Max codebook for the given dimension and bit-width.

    On the first call for a given ``(dim, bits)`` pair the codebook is
    computed via :func:`lloyd_max_codebook` and then cached for all
    subsequent calls.

    Parameters
    ----------
    dim : int
        Vector dimension.
    bits : int
        Bit-width per coordinate.

    Returns
    -------
    centroids : ndarray, shape (2**bits,)
    boundaries : ndarray, shape (2**bits - 1,)
    """
    return lloyd_max_codebook(dim, bits)
