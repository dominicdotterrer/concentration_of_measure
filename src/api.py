from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plane import orthonormalize, random_plane
from .cross_section import candidate_vertices, polygon_vertices, circumradius, inradius


@dataclass(frozen=True)
class CrossSectionResult:
    """All outputs for a single random cross-section draw."""
    basis: np.ndarray             # (2, n) orthonormal basis defining the plane
    vertices: np.ndarray          # (V, 2) polygon vertices (positive half)
    antipodal_vertices: np.ndarray  # (V, 2) = -vertices (negative half)
    inradius: float               # radius of largest inscribed disk
    circumradius: float           # radius of smallest enclosing disk
    payout: float                 # inradius / circumradius  (in (0, 1])


def cross_section(
    n: int | None = None,
    *,
    seed: int | None = None,
    matrix: np.ndarray | None = None,
) -> CrossSectionResult:
    """
    Compute a random 2D cross-section of the N-dimensional hypercube [-1, 1]^N.

    Exactly one of `n` or `matrix` must be supplied.

    Parameters
    ----------
    n      : dimension of the ambient space.  A random (2, n) Gaussian matrix
             is drawn (optionally seeded) and orthonormalised via Gram-Schmidt.
    seed   : optional RNG seed used only when `n` is supplied.
    matrix : a (2, n) matrix supplied by the caller.  Gram-Schmidt is applied
             to its rows before use, so the matrix need not be orthonormal.

    Returns
    -------
    CrossSectionResult with fields:
        basis               (2, n) orthonormal plane basis
        vertices            (V, 2) polygon vertices — positive half
        antipodal_vertices  (V, 2) = -vertices — negative half
        inradius            r  (largest inscribed disk radius)
        circumradius        R  (smallest enclosing disk radius)
        payout              r / R
    """
    if (n is None) == (matrix is None):
        raise ValueError("Provide exactly one of 'n' or 'matrix'.")

    if matrix is not None:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != 2:
            raise ValueError(
                f"'matrix' must have shape (2, n), got {matrix.shape}."
            )
        if matrix.shape[1] < 2:
            raise ValueError("'matrix' must have at least 2 columns.")
        basis = orthonormalize(matrix)
    else:
        basis = random_plane(n, seed=seed)

    cands = candidate_vertices(basis)
    verts = polygon_vertices(basis, cands)
    r = inradius(basis)
    R = circumradius(verts)

    return CrossSectionResult(
        basis=basis,
        vertices=verts,
        antipodal_vertices=-verts,
        inradius=r,
        circumradius=R,
        payout=r / R,
    )
