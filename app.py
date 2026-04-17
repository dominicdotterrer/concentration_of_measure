from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
import streamlit as st

from src.api import cross_section, CrossSectionResult

# ── colour palette ──────────────────────────────────────────────────────────
_BG           = "#F0F6FC"
_OUTER_DISK   = "#AED6F1"   # light blue    – circumscribed circle
_POLYGON      = "#A569BD"   # medium purple – cross-section polygon (translucent)
_POLYGON_ALPHA = 0.55
_INNER_DISK   = "#1A5276"   # dark blue     – inscribed circle (translucent)
_INNER_ALPHA  = 0.38
_LABEL_COLOR  = "#1A1A2E"

# ── figure builder ──────────────────────────────────────────────────────────

def _make_figure(result: CrossSectionResult) -> plt.Figure:
    """
    Render the cross-section polygon, inscribed disk, and circumscribed disk.

    Everything is normalised so the circumscribed circle has radius 1 in plot
    coordinates; only the inscribed circle's radius (= payout) varies.
    """
    # Combine positive and antipodal vertices, sort counter-clockwise.
    all_verts = np.vstack([result.vertices, result.antipodal_vertices])
    angles    = np.arctan2(all_verts[:, 1], all_verts[:, 0])
    scaled    = all_verts[np.argsort(angles)] / result.circumradius

    payout = result.payout

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.22, 1.18)
    ax.axis("off")

    ax.add_patch(Circle((0, 0), 1.0,    color=_OUTER_DISK,              zorder=1))
    ax.add_patch(MplPolygon(scaled,     color=_POLYGON,    closed=True,  alpha=_POLYGON_ALPHA, zorder=2))
    ax.add_patch(Circle((0, 0), payout, color=_INNER_DISK, alpha=_INNER_ALPHA, zorder=3))

    ax.text(
        0, -1.14,
        f"inradius / circumradius = {payout:.4f}",
        ha="center", va="top",
        fontsize=12, fontweight="bold", color=_LABEL_COLOR,
    )

    fig.tight_layout(pad=0.3)
    return fig


# ── page ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Concentration of Measure",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Concentration of Measure")
st.subheader("Random 2D cross-sections of the $N$-cube $[-1,1]^N$")

n = st.number_input(
    "Dimension **N**",
    min_value=3, max_value=1000, value=42, step=1,
    help="Ambient dimension. Try anything from 3 to a few hundred.",
)

if st.button(
    "🎲  Random cross-section of the N-cube",
    use_container_width=True,
    type="primary",
):
    st.session_state["result"] = cross_section(n=int(n))

if "result" in st.session_state:
    _, col, _ = st.columns([0.5, 7, 0.5])
    with col:
        st.pyplot(_make_figure(st.session_state["result"]), use_container_width=True)

# ── explanation ─────────────────────────────────────────────────────────────

st.divider()

st.markdown(r"""
### What am I looking at?

| Shape | Meaning |
|---|---|
| **Light-blue circle** | Circumscribed circle — smallest circle *enclosing* the cross-section |
| **Dark-purple polygon** | The actual cross-section of $[-1,1]^N$ with the random 2D plane |
| **Translucent inner circle** | Inscribed circle — largest circle *fitting inside* the cross-section |

The **inradius / circumradius ratio** (the *payout*) measures roundness.  A ratio of 1
means the cross-section is a perfect disk; smaller values indicate a more irregular polygon.
Both radii scale like $\sqrt{N / (2\log N)}$ for a random plane in $\mathbb{R}^N$, so the
absolute sizes grow with $N$ — the display is always normalised to the circumscribed radius.

---

### Why does it get rounder as $N$ grows?

This is the **concentration of measure** phenomenon.

> **Dvoretzky's theorem (1961):** Every infinite-dimensional Banach space contains
> almost-isometric copies of Euclidean space of arbitrarily high dimension.  For
> the hypercube, *almost every* 2D cross-section through the centre is approximately
> circular, with the approximation improving as $N \to \infty$.

The convergence is logarithmically slow — you need to push $N$ into the hundreds
before the roundness becomes visually striking.  But the direction of travel is
unmistakable: keep pressing the button at $N = 200$ and you will rarely see anything
very far from round.

**Why only $O(\log N)$ polygon edges?**
Each edge of the cross-section corresponds to one hypercube facet $x_k = \pm 1$ whose
perpendicular foot from the origin actually lands inside the polygon.  Whether foot $k$
is inside depends on how much each *other* column $j$ "sees" of column $k$.  A
Parseval-type identity pins this down exactly: if $\mathbf{c}_k$ denotes the $k$-th
column of the orthonormal basis, then

$$\sum_{j=1}^{N} \left(\frac{\mathbf{c}_j \cdot \mathbf{c}_k}{\|\mathbf{c}_k\|^2}\right)^{\!2} = \frac{1}{\|\mathbf{c}_k\|^2}.$$

For a *typical* column with $\|\mathbf{c}_k\|^2 \approx 2/N$, the right-hand side is
$N/2$, spread over $N$ terms — so most ratios have magnitude $\sim 1/\sqrt{2}$ and
many exceed 1, meaning foot $k$ is *outside* the polygon.  Only the $O(\log N)$
*extremal* columns — those whose norm is $\Theta(\sqrt{\log N / N})$ by extreme-value
theory — have a small enough right-hand side ($\approx N/(2\log N)$ spread over $N$
terms) that all ratios are safely below 1.

The upshot is a beautiful coincidence: Dvoretzky's theorem does **not** require many
polygon sides.  It requires that the few sides present are *well-placed*.  Because all
extremal column norms concentrate near the same value $\sqrt{2\log N / N}$, their feet
all land at radius $\approx\sqrt{N/(2\log N)}$ and spread nearly uniformly in angle —
making the $O(\log N)$-gon almost perfectly round.

Two celebrated consequences of the same underlying geometry:

**Johnson–Lindenstrauss lemma (1984).**  Any $n$ points in $\ell^2$ can be projected
down to $k = O(\log n \,/\, \varepsilon^2)$ dimensions while preserving *all* pairwise
distances to within a factor $(1 \pm \varepsilon)$.  The proof is essentially the
statement that random Gaussian projections concentrate — the same mechanism that
makes the cross-sections here round.

**Lindenstrauss: $\ell^2$ embeds almost isometrically into $\ell^1$.**  A random
linear map from $\ell^2^N$ into $\ell^1$ is nearly isometric with high probability —
almost every such map preserves Euclidean geometry.  Concretely, $\ell^2$ embeds into
$\ell^1$ with distortion that can be made arbitrarily close to 1 by taking the ambient
dimension large enough.  The cross-section you see here is the two-dimensional slice of
exactly this phenomenon: a random 2D plane "looks Euclidean" inside the $\ell^\infty$
ball (the hypercube), and the fidelity of that illusion grows with $N$.

---

*Increase $N$ and press the button repeatedly.  Even though any individual
cross-section can still look angular at moderate $N$, the* typical *cross-section
becomes progressively rounder — demonstrating that "most directions look the same"
in high dimensions.*
""")
