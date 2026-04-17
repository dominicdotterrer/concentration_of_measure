# Concentration of Measure

An interactive visualization of the **concentration of measure** phenomenon in high-dimensional geometry. Press a button, pick a dimension N, and watch random 2D cross-sections of the N-cube become nearly circular as N grows — a direct demonstration of **Dvoretzky's theorem**.

![screenshot placeholder](docs/screenshot.png)

## What it does

1. Samples a random 2D plane through the origin of ℝᴺ (two orthonormal Gaussian vectors via Gram–Schmidt).
2. Computes the cross-section of the hypercube \[-1, 1\]ᴺ with that plane — a centrally-symmetric convex polygon.
3. Finds the **inradius** r (largest inscribed disk) and **circumradius** R (smallest enclosing disk).
4. Displays the polygon, both disks, and the ratio r/R ("payout"). A ratio of 1 means a perfect circle.

The key insight: r/R → 1 as N → ∞ for almost every random plane. The convergence is logarithmically slow — push N into the hundreds to see it visually.

## Quickstart

### Prerequisites

- Python 3.10+
- `numpy`, `matplotlib`, `streamlit`

```bash
pip install numpy matplotlib streamlit
```

### Run the app

```bash
git clone <this-repo>
cd concentration_of_measure
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501), choose a dimension N, and click **Random cross-section of the N-cube**.

### Run the tests

```bash
python -m pytest tests/ -v
```

60 tests covering plane sampling, polygon geometry, inradius/circumradius, and the API.

## Repository layout

```
concentration_of_measure/
├── app.py                  # Streamlit UI
├── src/
│   ├── api.py              # cross_section() — top-level entry point
│   ├── plane.py            # orthonormalize(), random_plane()
│   └── cross_section.py    # candidate_vertices(), polygon_vertices(),
│                           #   circumradius(), inradius()
└── tests/
    ├── test_api.py
    ├── test_plane.py
    └── test_cross_section.py
```

## API

```python
from src.api import cross_section

result = cross_section(n=100)           # random plane in ℝ¹⁰⁰
result = cross_section(n=100, seed=42)  # reproducible
result = cross_section(matrix=M)        # caller-supplied (2, N) matrix
```

`CrossSectionResult` fields:

| Field | Type | Description |
|---|---|---|
| `basis` | `(2, N)` ndarray | orthonormal plane basis |
| `vertices` | `(V, 2)` ndarray | polygon vertices — positive half |
| `antipodal_vertices` | `(V, 2)` ndarray | `-vertices` — negative half |
| `inradius` | float | r |
| `circumradius` | float | R |
| `payout` | float | r / R |

## Mathematical background

**Dvoretzky's theorem (1961):** Every infinite-dimensional Banach space contains almost-isometric copies of Euclidean space of arbitrarily high dimension. For the hypercube, almost every 2D cross-section through the centre is approximately circular, with the approximation improving as N → ∞.

Two celebrated corollaries of the same geometry:

- **Johnson–Lindenstrauss lemma (1984):** n points in ℓ² can be projected to k = O(log n / ε²) dimensions while preserving all pairwise distances within (1 ± ε).
- **Lindenstrauss ℓ²→ℓ¹ embedding:** A random linear map from ℓ²ᴺ into ℓ¹ is nearly isometric with high probability.

## Planned extensions

- Cross-sections of general **Lₚ-balls** (p = 1, 2, ∞).
