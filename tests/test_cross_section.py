import numpy as np
import pytest
from src.cross_section import candidate_vertices, polygon_vertices, circumradius, inradius


def _rotation_basis(theta: float, n: int = 2) -> np.ndarray:
    """2×n basis whose first two columns are a 2D rotation by theta."""
    basis = np.zeros((2, n))
    basis[0, 0] = np.cos(theta)
    basis[1, 0] = -np.sin(theta)
    basis[0, 1] = np.sin(theta)
    basis[1, 1] = np.cos(theta)
    return basis


def _hexagon_basis() -> np.ndarray:
    """Orthonormal basis for the plane perpendicular to [1,1,1] in R^3.
    The cross-section of [-1,1]^3 by this plane is a regular hexagon."""
    v1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v2 = np.array([1.0,  1.0, -2.0]) / np.sqrt(6)
    return np.stack([v1, v2])


# ---------------------------------------------------------------------------
# candidate_vertices tests
# ---------------------------------------------------------------------------

def test_identity_n2():
    # N=2 identity: one constraint pair, two sign combos -> 2 candidates
    basis = np.eye(2)
    pts = candidate_vertices(basis)
    assert pts.shape == (2, 2)
    assert any(np.allclose(p, [1.0, 1.0]) for p in pts)   # (+1,+1)
    assert any(np.allclose(p, [1.0, -1.0]) for p in pts)  # (+1,-1)


def test_rotation_45_n2():
    # 45-degree rotation: (+1,+1) intersection is (sqrt(2), 0)
    basis = _rotation_basis(np.pi / 4)
    pts = candidate_vertices(basis)
    assert pts.shape == (2, 2)
    assert any(np.allclose(p, [np.sqrt(2), 0.0], atol=1e-12) for p in pts)


def test_count_general():
    # For an n-dim basis: C(n,2) pairs × 2 sign combos = n*(n-1) candidates
    from src.plane import random_plane
    for n in [3, 5, 10, 20]:
        basis = random_plane(n, seed=n)
        pts = candidate_vertices(basis)
        assert pts.shape[0] == n * (n - 1)
        assert pts.shape[1] == 2


def test_parallel_lines_skipped():
    basis = np.array([[1.0, 1.0], [0.0, 0.0]])
    pts = candidate_vertices(basis)
    assert pts.shape == (0, 2)


# ---------------------------------------------------------------------------
# polygon_vertices tests
# ---------------------------------------------------------------------------

def test_filter_identity_n2():
    # N=2 identity: both candidates (1,1) and (1,-1) satisfy all constraints
    basis = np.eye(2)
    cands = candidate_vertices(basis)
    verts = polygon_vertices(basis, cands)
    assert verts.shape == (2, 2)


def test_filter_satisfies_constraint():
    from src.plane import random_plane
    for n, seed in [(5, 0), (10, 1), (50, 2)]:
        basis = random_plane(n, seed=seed)
        verts = polygon_vertices(basis, candidate_vertices(basis))
        lifts = verts @ basis
        assert np.all(np.abs(lifts) <= 1.0 + 1e-9), f"vertex outside hypercube for n={n}"


def test_filter_two_coords_tight():
    from src.plane import random_plane
    basis = random_plane(20, seed=3)
    verts = polygon_vertices(basis, candidate_vertices(basis))
    for v in verts:
        tight = np.sum(np.abs(np.abs(v @ basis) - 1.0) < 1e-9)
        assert tight >= 2, f"vertex {v} has fewer than 2 tight constraints"


def test_filter_reduces_candidates():
    from src.plane import random_plane
    basis = random_plane(10, seed=0)
    cands = candidate_vertices(basis)
    verts = polygon_vertices(basis, cands)
    assert len(verts) < len(cands)


def test_filter_no_negatives():
    from src.plane import random_plane
    basis = random_plane(15, seed=99)
    verts = polygon_vertices(basis, candidate_vertices(basis))
    for i, v in enumerate(verts):
        for j, w in enumerate(verts):
            if i != j:
                assert not np.allclose(v, -w, atol=1e-9), \
                    "antipodal pair found — negatives should not appear until reflection step"


def test_filter_empty_input():
    basis = np.eye(2)
    result = polygon_vertices(basis, np.empty((0, 2)))
    assert result.shape == (0, 2)


def test_hexagon_vertices():
    # Regular hexagon: all 6 vertices are mixed-sign (+1,-1 or -1,+1).
    # The positive-half should give exactly 3 vertices (one per constraint pair).
    basis = _hexagon_basis()
    verts = polygon_vertices(basis, candidate_vertices(basis))
    assert len(verts) == 3
    # All norms should equal sqrt(2) (hexagon circumradius)
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), np.sqrt(2), atol=1e-12)


# ---------------------------------------------------------------------------
# circumradius tests
# ---------------------------------------------------------------------------

def test_circumradius_identity_n2():
    basis = np.eye(2)
    verts = polygon_vertices(basis, candidate_vertices(basis))
    assert np.isclose(circumradius(verts), np.sqrt(2))


def test_circumradius_hexagon():
    basis = _hexagon_basis()
    verts = polygon_vertices(basis, candidate_vertices(basis))
    assert np.isclose(circumradius(verts), np.sqrt(2), atol=1e-12)


def test_circumradius_is_max_norm():
    from src.plane import random_plane
    basis = random_plane(20, seed=5)
    verts = polygon_vertices(basis, candidate_vertices(basis))
    R = circumradius(verts)
    assert np.isclose(R, np.max(np.linalg.norm(verts, axis=1)))


# ---------------------------------------------------------------------------
# inradius tests
# ---------------------------------------------------------------------------

def test_inradius_identity_n2():
    # Square [-1,1]^2: distance from origin to each face is 1
    basis = np.eye(2)
    assert np.isclose(inradius(basis), 1.0)


def test_inradius_hexagon():
    # Regular hexagon: inradius = circumradius * sqrt(3)/2
    basis = _hexagon_basis()
    verts = polygon_vertices(basis, candidate_vertices(basis))
    np.testing.assert_allclose(inradius(basis), circumradius(verts) * np.sqrt(3) / 2, atol=1e-12)


def test_inradius_leq_circumradius():
    from src.plane import random_plane
    for n, seed in [(5, 0), (10, 1), (20, 2), (50, 3)]:
        basis = random_plane(n, seed=seed)
        r = inradius(basis)
        R = circumradius(polygon_vertices(basis, candidate_vertices(basis)))
        assert r <= R + 1e-9, f"inradius > circumradius for n={n}"


def test_inradius_positive():
    from src.plane import random_plane
    for n, seed in [(5, 0), (10, 1), (20, 2)]:
        basis = random_plane(n, seed=seed)
        assert inradius(basis) > 0


def test_inradius_increases_with_dimension():
    # Concentration of measure: r/R should increase monotonically with N,
    # converging toward 1 (logarithmically slowly by Dvoretzky's theorem).
    from src.plane import random_plane
    ns = [10, 50, 200]
    ratios = []
    for n in ns:
        basis = random_plane(n, seed=42)
        verts = polygon_vertices(basis, candidate_vertices(basis))
        ratios.append(inradius(basis) / circumradius(verts))
    assert all(r2 > r1 for r1, r2 in zip(ratios, ratios[1:])), \
        f"r/R not increasing: N={ns}, r/R={[f'{r:.4f}' for r in ratios]}"
