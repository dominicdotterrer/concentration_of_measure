import numpy as np
import pytest
from src.api import cross_section, CrossSectionResult


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_requires_exactly_one_arg():
    with pytest.raises(ValueError):
        cross_section()                                  # neither
    with pytest.raises(ValueError):
        cross_section(n=5, matrix=np.eye(2)[:, :5])     # both


def test_matrix_shape_validation():
    with pytest.raises(ValueError):
        cross_section(matrix=np.ones((3, 5)))   # wrong first dimension
    with pytest.raises(ValueError):
        cross_section(matrix=np.ones((2, 1)))   # too few columns


# ---------------------------------------------------------------------------
# return type and shape
# ---------------------------------------------------------------------------

def test_returns_dataclass():
    result = cross_section(n=10, seed=0)
    assert isinstance(result, CrossSectionResult)


def test_basis_shape_and_orthonormality():
    result = cross_section(n=20, seed=1)
    assert result.basis.shape == (2, 20)
    norms = np.linalg.norm(result.basis, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.basis[0] @ result.basis[1], 0.0, atol=1e-12)


def test_vertices_shape():
    result = cross_section(n=10, seed=2)
    assert result.vertices.ndim == 2
    assert result.vertices.shape[1] == 2
    assert result.antipodal_vertices.shape == result.vertices.shape


def test_antipodal_vertices_are_negatives():
    result = cross_section(n=15, seed=3)
    np.testing.assert_array_equal(result.antipodal_vertices, -result.vertices)


def test_payout_equals_ratio():
    result = cross_section(n=10, seed=4)
    assert np.isclose(result.payout, result.inradius / result.circumradius)


def test_payout_in_range():
    for seed in range(5):
        result = cross_section(n=10, seed=seed)
        assert 0 < result.payout <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    r1 = cross_section(n=30, seed=99)
    r2 = cross_section(n=30, seed=99)
    np.testing.assert_array_equal(r1.basis, r2.basis)
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    assert r1.payout == r2.payout


def test_different_seeds_differ():
    r1 = cross_section(n=30, seed=0)
    r2 = cross_section(n=30, seed=1)
    assert not np.allclose(r1.basis, r2.basis)


# ---------------------------------------------------------------------------
# matrix path
# ---------------------------------------------------------------------------

def test_matrix_path_runs():
    m = np.random.default_rng(7).normal(size=(2, 10))
    result = cross_section(matrix=m)
    assert result.basis.shape == (2, 10)


def test_matrix_path_orthonormalises():
    # Supply a non-orthonormal matrix; rows of basis should be orthonormal.
    m = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
    result = cross_section(matrix=m)
    norms = np.linalg.norm(result.basis, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.basis[0] @ result.basis[1], 0.0, atol=1e-12)


def test_matrix_path_seed_ignored():
    # seed kwarg has no effect when matrix is supplied (matrix is deterministic)
    m = np.random.default_rng(0).normal(size=(2, 10))
    r1 = cross_section(matrix=m)
    r2 = cross_section(matrix=m, seed=42)
    np.testing.assert_array_equal(r1.basis, r2.basis)


def test_hexagon_via_matrix():
    # Regular hexagon: payout = sqrt(3)/2
    v1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v2 = np.array([1.0,  1.0, -2.0]) / np.sqrt(6)
    result = cross_section(matrix=np.stack([v1, v2]))
    np.testing.assert_allclose(result.payout, np.sqrt(3) / 2, atol=1e-12)


# ---------------------------------------------------------------------------
# cyclic-curve family
# ---------------------------------------------------------------------------
# The cyclic-curve matrix is defined by two geometric progressions:
#   row 0: [a,  a^2,  ..., a^n]
#   row 1: [b,  b^2,  ..., b^n]   with 0 < a < b < 1
#
# After Gram-Schmidt this defines a specific 2D plane.  It is a useful
# parametric family for testing because the entries decay at controlled rates
# and the resulting cross-sections span a range of shapes.

@pytest.mark.parametrize("a,b,n", [
    (1/3, 2/3, 10),
    (1/3, 2/3, 30),
    (0.1, 0.9, 15),
    (0.4, 0.6, 20),
])
def test_cyclic_curve_payout_valid(a, b, n):
    idx = np.arange(1, n + 1)
    matrix = np.stack([a ** idx, b ** idx])
    result = cross_section(matrix=matrix)
    assert 0 < result.payout <= 1.0 + 1e-9


@pytest.mark.parametrize("a,b,n", [
    (1/3, 2/3, 10),
    (0.2, 0.8, 20),
])
def test_cyclic_curve_vertices_in_cube(a, b, n):
    idx = np.arange(1, n + 1)
    result = cross_section(matrix=np.stack([a ** idx, b ** idx]))
    lifts = result.vertices @ result.basis
    np.testing.assert_array_less(np.abs(lifts), 1.0 + 1e-9)


def test_cyclic_curve_antipodal():
    a, b, n = 1/3, 2/3, 10
    idx = np.arange(1, n + 1)
    result = cross_section(matrix=np.stack([a ** idx, b ** idx]))
    np.testing.assert_array_equal(result.antipodal_vertices, -result.vertices)


def test_cyclic_curve_payout_converges():
    # Geometric entries decay exponentially, so the polygon shape is essentially
    # fixed once n is large enough to capture the non-negligible terms.
    # Payout for n=30 and n=50 should be indistinguishable.
    a, b = 1/3, 2/3
    payouts = []
    for n in [20, 30, 50]:
        idx = np.arange(1, n + 1)
        payouts.append(cross_section(matrix=np.stack([a ** idx, b ** idx])).payout)
    assert abs(payouts[-1] - payouts[-2]) < 0.01, \
        f"payout did not converge: {payouts}"


def test_cyclic_curve_no_convergence_for_a_b_gt_1():
    # For a, b > 1 entries grow with index, so later dimensions keep shifting the
    # effective 2D plane — the polygon does NOT stabilize the way it does for a, b < 1.
    # Contrast: the a,b < 1 convergence test asserts |payout(30) - payout(50)| < 0.01;
    # here we assert the analogous gap is still > 0.01 (not yet converged).
    a, b = 1.1, 1.3
    payouts = []
    for n in [10, 20, 50]:
        idx = np.arange(1, n + 1)
        payouts.append(cross_section(matrix=np.stack([a ** idx, b ** idx])).payout)
    assert abs(payouts[-1] - payouts[-2]) > 0.01, \
        f"expected non-convergence for a={a}, b={b}; got payouts={payouts}"
