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


def _fourier_sine_matrix(n: int, k_a: int, k_b: int) -> np.ndarray:
    """Unnormalized Fourier sine basis; orthonormalization is left to the caller/API."""
    j = np.arange(n, dtype=float)
    return np.array([
        np.sin(2 * np.pi * k_a * j / n),
        np.sin(2 * np.pi * k_b * j / n),
    ])


def test_fourier_diagonal_orthogonality():
    # sum_j sin(2π*k*j/N) = 0 for integer k >= 1: both rows are perpendicular to
    # the all-ones cube diagonal, so the cross-section plane passes through the
    # cube's centroid with no component along the main diagonal.
    for n, k_a, k_b in [(6, 1, 2), (8, 1, 3), (12, 1, 5), (50, 3, 7)]:
        M = _fourier_sine_matrix(n, k_a, k_b)
        np.testing.assert_allclose(M.sum(axis=1), 0.0, atol=1e-12,
                                   err_msg=f"n={n}, k_a={k_a}, k_b={k_b}")


def test_fourier_n6_k1_k2_square():
    # N=6, frequencies 1 and 2: after Gram-Schmidt the 4 non-zero columns all have
    # the same norm (1/sqrt(2)), so the constraints reduce to x+y=±2 and x-y=±2 —
    # a square with inradius sqrt(2), circumradius 2, payout 1/sqrt(2).
    #
    # Note: the Fourier basis has antisymmetric columns (col_{N-k} = -col_k), which
    # causes candidate_vertices to include both a vertex and its antipode in the
    # "positive half" output (a structural limitation for symmetric bases only).
    # The scalar quantities — inradius, circumradius, payout — are unaffected.
    from src.api import cross_section
    result = cross_section(matrix=_fourier_sine_matrix(6, 1, 2))
    np.testing.assert_allclose(result.inradius,    np.sqrt(2),        atol=1e-12)
    np.testing.assert_allclose(result.circumradius, 2.0,              atol=1e-12)
    np.testing.assert_allclose(result.payout,      1.0 / np.sqrt(2), atol=1e-10)

    # Unique vertices across both halves form exactly the 4 corners of the square.
    all_verts = np.vstack([result.vertices, result.antipodal_vertices])
    unique_verts = np.unique(np.round(all_verts, 9), axis=0)
    assert len(unique_verts) == 4, f"Expected 4 unique square corners, got {len(unique_verts)}"
    np.testing.assert_allclose(np.sort(np.linalg.norm(unique_verts, axis=1)),
                                [2.0, 2.0, 2.0, 2.0], atol=1e-9)


def test_vertex_count_grows_with_dimension():
    # The cross-section polygon has O(log N) vertices by extreme-value theory on
    # column norms: only the ~O(log N) columns with above-average norm produce
    # valid (inside-hypercube) vertex candidates.  Even so, Dvoretzky holds
    # because those vertices are nearly uniformly spread on the circumscribed
    # circle — a 14-gon with evenly spaced vertices has payout cos(π/14) ≈ 0.975.
    # We verify the mean count (positive half) grows across 20 random planes.
    from src.plane import random_plane
    ns = [10, 50, 200]
    n_trials = 20

    mean_counts = []
    for n in ns:
        counts = []
        for s in range(n_trials):
            basis = random_plane(n, seed=s)
            counts.append(len(polygon_vertices(basis, candidate_vertices(basis))))
        mean_counts.append(float(np.mean(counts)))

    assert all(c2 > c1 for c1, c2 in zip(mean_counts, mean_counts[1:])), (
        f"Mean vertex count not growing with N: "
        f"N={ns}, mean_counts={[f'{c:.1f}' for c in mean_counts]}"
    )
