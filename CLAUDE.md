# concentration_of_measure — Backend

## Purpose

Backend logic for a web game demonstrating the **concentration of measure** phenomenon in high-dimensional geometry. The frontend (visualization, UI) lives in a separate repository; this repo exposes the mathematical computations as a callable API or library.

## The Game

1. The user selects a dimension **N** and clicks "Random cross-section."
2. The backend generates a random 2D orthonormal basis in ℝᴺ (two orthonormal vectors spanning a 2D plane through the origin).
3. It computes the 2D cross-section of the N-dimensional hypercube \([-1,1]^N\) defined by that plane — a centrally-symmetric convex polygon.
4. It computes:
   - **r** — the inradius (radius of the largest inscribed disk centered at origin)
   - **R** — the circumradius (radius of the smallest enclosing disk centered at origin)
   - **payout** = r / R  (ranges from 0 to 1; equals 1 for a perfect circle)
5. It returns the polygon vertices (and their antipodes), both disk radii, and the payout ratio to the frontend.

The key insight: as N → ∞, payout → 1 for almost every random cross-section — a manifestation of concentration of measure / Dvoretzky's theorem. The convergence is logarithmically slow.

## Planned Extensions

- Support cross-sections of general **Lₚ-balls** (p = 1, 2, ∞ are the natural targets), not just the hypercube (L∞-ball).

## Language & Stack

- **Python** (see `.gitignore` — standard Python project conventions)
- Numerical work: `numpy`
- API layer: TBD (FastAPI is the likely choice for serving the frontend)

## Core Mathematical Operations

| Operation | Description |
|---|---|
| Random 2D plane | Sample two orthonormal row vectors in ℝᴺ via Gram-Schmidt on random Gaussian vectors |
| Cross-section polygon | Intersect [-1,1]^N with the 2D plane; result is a centrally-symmetric convex polygon |
| Inradius **r** | `1 / max_k ‖basis[:,k]‖`, minimised over constraint lines whose feet lie inside the polygon |
| Circumradius **R** | `max ‖v‖` over all polygon vertices |
| Payout | r / R |

## Repository Structure

```
concentration_of_measure/
├── CLAUDE.md
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── api.py            # cross_section() — top-level entry point
│   ├── plane.py          # orthonormalize(), random_plane()
│   └── cross_section.py  # candidate_vertices(), polygon_vertices(),
│                         # circumradius(), inradius()
└── tests/
    ├── test_api.py
    ├── test_plane.py
    └── test_cross_section.py
```

## Modules

### `src/api.py` — top-level entry point

```python
result = cross_section(n=100)                   # random plane in R^100
result = cross_section(n=100, seed=42)          # reproducible
result = cross_section(matrix=my_2xN_array)     # caller-supplied matrix (Gram-Schmidt applied)
```

Returns a frozen `CrossSectionResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `basis` | `(2, n)` ndarray | orthonormal plane basis |
| `vertices` | `(V, 2)` ndarray | polygon vertices — positive half |
| `antipodal_vertices` | `(V, 2)` ndarray | `-vertices` — negative half |
| `inradius` | float | r |
| `circumradius` | float | R |
| `payout` | float | r / R |

### `src/plane.py`

`orthonormalize(v) -> np.ndarray` — Gram-Schmidt on a `(2, n)` matrix: normalise row 0, subtract its projection from row 1, normalise row 1.

`random_plane(n, seed=None) -> np.ndarray` — draw i.i.d. N(0,1) entries, then call `orthonormalize`.

### `src/cross_section.py`

`candidate_vertices(basis) -> np.ndarray` — for each pair of constraint dimensions (i, j) solve **both** sign combinations (+1/+1 and +1/−1) via Cramer's rule, yielding up to N(N−1) candidate 2D points. The (−1/+1) and (−1/−1) solutions are their negatives and are recovered by reflection after filtering.

`polygon_vertices(basis, candidates, tol=1e-9) -> np.ndarray` — keep only candidates whose N-dimensional lift `candidates @ basis` lies entirely in `[-1-tol, 1+tol]^N`.

`circumradius(vertices) -> float` — `max ‖v‖` over the positive-half vertex set.

`inradius(basis, tol=1e-9) -> float` — for each constraint k, the foot of perpendicular from origin to the line `basis[0,k]*x + basis[1,k]*y = 1` is `basis[:,k] / ‖basis[:,k]‖²`, at distance `1/‖basis[:,k]‖`. The inradius is the minimum such distance over the constraints whose feet lie inside the polygon.

## Running Tests

```bash
python -m pytest tests/ -v
```
