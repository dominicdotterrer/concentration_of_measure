import numpy as np
from itertools import combinations  # used by candidate_vertices


def candidate_vertices(basis: np.ndarray) -> np.ndarray:
    """
    Compute candidate polygon vertices from all pairwise sign combinations.

    Each constraint k defines two boundary lines: a_k*x + b_k*y = +/-1.
    For every pair of constraints (i, j) we solve both sign combinations:

      (+1, +1): [[a_i,b_i],[a_j,b_j]] [x,y]^T = [1, 1]^T
      (+1, -1): [[a_i,b_i],[a_j,b_j]] [x,y]^T = [1,-1]^T

    Both are solved via Cramer's rule with the same determinant.  The (-1,+1)
    and (-1,-1) solutions are the negatives of these and are recovered later by
    the antipodal reflection step, so they are not computed here.

    Parameters
    ----------
    basis : (2, n) orthonormal matrix returned by random_plane()

    Returns
    -------
    pts : (K, 2) array, K <= N*(N-1)
    """
    n = basis.shape[1]
    candidates = []

    for i, j in combinations(range(n), 2):
        a_i, b_i = basis[0, i], basis[1, i]
        a_j, b_j = basis[0, j], basis[1, j]

        det = a_i * b_j - a_j * b_i
        if abs(det) < 1e-12:   # parallel lines — no finite intersection
            continue

        # (+1, +1): rhs = [1, 1]
        candidates.append([(b_j - b_i) / det, (a_i - a_j) / det])

        # (+1, -1): rhs = [1, -1]
        candidates.append([(b_j + b_i) / det, -(a_i + a_j) / det])

    return np.array(candidates) if candidates else np.empty((0, 2))


def polygon_vertices(basis: np.ndarray, candidates: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Filter candidate vertices to those that lie within the hypercube [-1, 1]^N.

    Each candidate p is lifted to R^N via basis.T @ p = candidates @ basis (row-wise).
    A point is a true polygon vertex iff every coordinate of its lift is in [-1, 1];
    by construction at least two coordinates are exactly +-1.

    Parameters
    ----------
    basis      : (2, n) orthonormal basis matrix
    candidates : (K, 2) array from candidate_vertices()
    tol        : numerical tolerance added to the <=1 bound

    Returns
    -------
    vertices : (V, 2) array of actual polygon vertices
    """
    if len(candidates) == 0:
        return candidates

    lifts = candidates @ basis          # (K, N): row k is the N-dim lift of candidate k
    mask = np.all(np.abs(lifts) <= 1.0 + tol, axis=1)
    return candidates[mask]


def circumradius(vertices: np.ndarray) -> float:
    """
    Radius of the smallest enclosing disk of the cross-section polygon.

    The polygon is centered at the origin, so the enclosing disk is also
    centered there and its radius equals the maximum vertex norm.  The
    half-set of positive vertices suffices because negatives have equal norms.

    Parameters
    ----------
    vertices : (V, 2) array from polygon_vertices()

    Returns
    -------
    R : float
    """
    return float(np.max(np.linalg.norm(vertices, axis=1)))


def inradius(basis: np.ndarray, tol: float = 1e-9) -> float:
    """
    Radius of the largest inscribed disk of the cross-section polygon.

    For each constraint k the boundary line is a_k*x + b_k*y = 1, where
    [a_k, b_k] = basis[:, k].  The foot of the perpendicular from the origin
    to this line is:

        foot_k = [a_k, b_k] / (a_k^2 + b_k^2)

    at distance 1 / sqrt(a_k^2 + b_k^2).  The foot is a valid edge point iff
    its N-dim lift lies entirely within [-1, 1]^N (the coordinate for
    constraint k is automatically 1 by construction).  The inradius is the
    minimum such distance over all valid constraints.

    The -1 boundary lines are symmetric and yield the same distances, so only
    the +1 lines need be checked.

    Parameters
    ----------
    basis : (2, n) orthonormal basis matrix
    tol   : numerical tolerance for the inside-hypercube check

    Returns
    -------
    r : float
    """
    col_norms_sq = np.sum(basis ** 2, axis=0)         # (N,)
    feet = basis.T / col_norms_sq[:, None]             # (N, 2): foot_k = col_k / ||col_k||^2
    lifts = feet @ basis                               # (N, N): row k is the N-dim lift of foot_k
    inside = np.all(np.abs(lifts) <= 1.0 + tol, axis=1)

    if not np.any(inside):
        return np.inf

    # distance_k = 1 / sqrt(col_norms_sq[k])
    # min distance corresponds to max col_norms_sq among valid constraints
    return float(1.0 / np.sqrt(np.max(col_norms_sq[inside])))
