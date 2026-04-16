import numpy as np
import pytest
from src.plane import random_plane


@pytest.mark.parametrize("n", [2, 3, 10, 100, 1000])
def test_shape(n):
    basis = random_plane(n, seed=0)
    assert basis.shape == (2, n)


@pytest.mark.parametrize("n", [2, 3, 10, 100, 1000])
def test_unit_rows(n):
    basis = random_plane(n, seed=1)
    norms = np.linalg.norm(basis, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 10, 100, 1000])
def test_orthogonality(n):
    basis = random_plane(n, seed=2)
    dot = np.dot(basis[0], basis[1])
    np.testing.assert_allclose(dot, 0.0, atol=1e-12)


def test_reproducibility():
    b1 = random_plane(50, seed=42)
    b2 = random_plane(50, seed=42)
    np.testing.assert_array_equal(b1, b2)


def test_different_seeds_differ():
    b1 = random_plane(50, seed=0)
    b2 = random_plane(50, seed=1)
    assert not np.allclose(b1, b2)


def test_no_seed_runs():
    # should not raise
    random_plane(10)
