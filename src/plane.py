import numpy as np


def orthonormalize(v: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt to a (2, n) matrix, returning a (2, n) matrix whose
    rows are orthonormal.  The first row is normalised directly; the second is
    made orthogonal to the first and then normalised.
    """
    u0 = v[0] / np.linalg.norm(v[0])
    u1 = v[1] - np.dot(v[1], u0) * u0
    u1 = u1 / np.linalg.norm(u1)
    return np.stack([u0, u1])


def random_plane(n: int, seed: int | None = None) -> np.ndarray:
    """
    Sample a random 2D plane in R^n.

    Returns a (2, n) matrix whose rows are an orthonormal basis for the plane,
    constructed by drawing a (2, n) Gaussian matrix and applying Gram-Schmidt
    to the two row vectors.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, size=(2, n))
    return orthonormalize(v)
