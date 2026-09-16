"""Arena geometry and vectorised 2-D ray casting.

Geometry is taken from ``ros_gz_sim_demos/worlds/world.sdf``:

* four walls centred at x,y = +-4.0 with thickness 0.1  -> inner faces at +-3.95
* four cylindrical pillars of radius 0.20 at
  (-1.0, 1.5), (1.3, 1.0), (-1.4, -0.8), (0.8, -1.6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

WALL_HALF = 4.0
WALL_THICKNESS = 0.1
INNER_HALF = WALL_HALF - WALL_THICKNESS / 2.0  # 3.95

PILLAR_RADIUS = 0.20
PILLARS: List[Tuple[float, float]] = [(-1.0, 1.5), (1.3, 1.0), (-1.4, -0.8), (0.8, -1.6)]


@dataclass
class Arena:
    """Axis-aligned square arena with circular pillars."""

    half_size: float = INNER_HALF
    pillars: Sequence[Tuple[float, float]] = field(default_factory=lambda: list(PILLARS))
    pillar_radius: float = PILLAR_RADIUS

    def __post_init__(self) -> None:
        h = self.half_size
        # wall inner faces as segments (p, q)
        self.seg_p = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
        self.seg_q = np.array([[h, -h], [h, h], [-h, h], [-h, -h]], dtype=np.float64)
        self.seg_r = self.seg_q - self.seg_p  # direction vectors
        self.circ_c = np.asarray(self.pillars, dtype=np.float64).reshape(-1, 2)
        self.circ_r = np.full(len(self.circ_c), self.pillar_radius, dtype=np.float64)

    # ------------------------------------------------------------------ queries
    def clearance(self, point: np.ndarray) -> float:
        """Distance from ``point`` to the nearest obstacle surface (walls or pillars).

        Negative if the point is inside a wall or pillar.
        """
        x, y = float(point[0]), float(point[1])
        wall = self.half_size - max(abs(x), abs(y))
        if len(self.circ_c):
            d = np.hypot(self.circ_c[:, 0] - x, self.circ_c[:, 1] - y) - self.circ_r
            pillar = float(d.min())
        else:
            pillar = np.inf
        return min(wall, pillar)

    def is_free(self, point: np.ndarray, radius: float) -> bool:
        return self.clearance(point) > radius

    def sample_free(
        self,
        rng: np.random.Generator,
        radius: float,
        limit: float = 3.2,
        max_tries: int = 200,
    ) -> np.ndarray:
        """Uniformly sample a point in [-limit, limit]^2 with ``clearance > radius``."""
        for _ in range(max_tries):
            p = rng.uniform(-limit, limit, size=2)
            if self.is_free(p, radius):
                return p
        return np.zeros(2)

    # --------------------------------------------------------------- ray casting
    def ray_cast(self, origin: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Return the range along each world-frame ``angle`` from ``origin``.

        Rays that hit nothing within ``max_range`` return ``max_range``.
        Fully vectorised: (n_rays x n_segments) and (n_rays x n_circles).
        """
        ox, oy = float(origin[0]), float(origin[1])
        dx = np.cos(angles)
        dy = np.sin(angles)
        t_best = np.full(angles.shape, max_range, dtype=np.float64)

        # --- segments: o + t d = p + u r
        px = self.seg_p[None, :, 0] - ox  # (1, S)
        py = self.seg_p[None, :, 1] - oy
        rx = self.seg_r[None, :, 0]
        ry = self.seg_r[None, :, 1]
        dxs = dx[:, None]
        dys = dy[:, None]
        denom = dxs * ry - dys * rx  # d x r
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (px * ry - py * rx) / denom
            u = (px * dys - py * dxs) / denom
        valid = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)
        t_best = np.minimum(t_best, t.min(axis=1))

        # --- circles: |o + t d - c|^2 = R^2
        if len(self.circ_c):
            fx = ox - self.circ_c[None, :, 0]  # (1, C)
            fy = oy - self.circ_c[None, :, 1]
            b = fx * dxs + fy * dys  # (N, C)
            c0 = fx * fx + fy * fy - self.circ_r[None, :] ** 2
            disc = b * b - c0
            hit = disc >= 0.0
            sq = np.sqrt(np.where(hit, disc, 0.0))
            t1 = -b - sq
            t2 = -b + sq
            t_c = np.where(t1 >= 0.0, t1, t2)
            t_c = np.where(hit & (t_c >= 0.0), t_c, np.inf)
            t_best = np.minimum(t_best, t_c.min(axis=1))

        return t_best

    # ------------------------------------------------------------------ drawing
    def draw(self, ax, wall_color: str = "black", pillar_color: str = "dimgray") -> None:
        """Draw the arena on a matplotlib axis."""
        import matplotlib.patches as patches

        h = self.half_size
        ax.add_patch(
            patches.Rectangle(
                (-WALL_HALF - WALL_THICKNESS / 2, -WALL_HALF - WALL_THICKNESS / 2),
                2 * WALL_HALF + WALL_THICKNESS,
                2 * WALL_HALF + WALL_THICKNESS,
                fill=False,
                lw=6,
                ec=wall_color,
            )
        )
        for (cx, cy), r in zip(self.circ_c, self.circ_r):
            ax.add_patch(patches.Circle((cx, cy), r, fc=pillar_color, ec="black"))
        ax.set_xlim(-h - 0.4, h + 0.4)
        ax.set_ylim(-h - 0.4, h + 0.4)
        ax.set_aspect("equal")


DEFAULT_ARENA = Arena()
