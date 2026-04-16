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
