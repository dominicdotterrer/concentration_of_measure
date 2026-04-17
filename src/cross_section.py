import numpy as np

_BATCH = 8192   # rows processed at a time in polygon_vertices


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
    i_idx, j_idx = np.triu_indices(n, k=1)   # all (i,j) pairs with i < j

    a_i, b_i = basis[0, i_idx], basis[1, i_idx]
    a_j, b_j = basis[0, j_idx], basis[1, j_idx]

    det = a_i * b_j - a_j * b_i
    valid = np.abs(det) > 1e-12
    if not np.any(valid):
        return np.empty((0, 2))

    a_i, b_i = a_i[valid], b_i[valid]
    a_j, b_j = a_j[valid], b_j[valid]
    det = det[valid]

    # (+1, +1): x = (b_j - b_i)/det,  y = (a_i - a_j)/det
    x_pp = (b_j - b_i) / det
    y_pp = (a_i - a_j) / det

    # (+1, -1): x = (b_j + b_i)/det,  y = -(a_i + a_j)/det
    x_pm = (b_j + b_i) / det
    y_pm = -(a_i + a_j) / det

    pp = np.stack([x_pp, y_pp], axis=1)
    pm = np.stack([x_pm, y_pm], axis=1)
    return np.concatenate([pp, pm], axis=0)


def polygon_vertices(basis: np.ndarray, candidates: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Filter candidate vertices to those that lie within the hypercube [-1, 1]^N.

    Each candidate p is lifted to R^N via basis.T @ p = candidates @ basis (row-wise).
    A point is a true polygon vertex iff every coordinate of its lift is in [-1, 1];
    by construction at least two coordinates are exactly +-1.

    Candidates are processed in batches of _BATCH rows to bound peak memory use.

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

    keep = []
    for start in range(0, len(candidates), _BATCH):
        chunk = candidates[start : start + _BATCH]
        lifts = chunk @ basis                            # (_BATCH, N)
        mask = np.all(np.abs(lifts) <= 1.0 + tol, axis=1)
        keep.append(chunk[mask])

    return np.concatenate(keep, axis=0) if keep else np.empty((0, 2))


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
    active = col_norms_sq > tol                        # skip zero-norm columns (0*x+0*y=±1 is never tight)
    if not np.any(active):
        return np.inf

    active_norms_sq = col_norms_sq[active]
    feet = basis[:, active].T / active_norms_sq[:, None]  # (K, 2): foot_k = col_k / ||col_k||^2
    lifts = feet @ basis                                   # (K, N): full lift; zero-norm coords are 0
    inside = np.all(np.abs(lifts) <= 1.0 + tol, axis=1)

    if not np.any(inside):
        return np.inf

    # distance_k = 1 / sqrt(col_norms_sq[k])
    # min distance corresponds to max col_norms_sq among valid constraints
    return float(1.0 / np.sqrt(np.max(active_norms_sq[inside])))
