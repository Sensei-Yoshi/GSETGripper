"""Diagnostic overlay for the projected two-pad contact fraction."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from contact_area import ContactEstimate


def plot_estimate(est: ContactEstimate, path: Path, title: str) -> None:
    b = est.boundary
    fig, (ax, axk) = plt.subplots(
        2, 1, figsize=(7.5, 9.5), gridspec_kw={"height_ratios": [3.2, 1]}
    )

    closed = np.vstack((b.pts, b.pts[:1]))
    ax.plot(closed[:, 0], closed[:, 1], color="0.55", lw=1.0, zorder=1)
    bad = ~est.reachable
    if bad.any():
        ax.scatter(
            b.pts[bad, 0], b.pts[bad, 1], s=3, color="crimson", alpha=0.5,
            zorder=2, label="curvature/concavity unreachable",
        )

    pad_lo, pad_hi = est.pad_band
    ax.axhline(pad_lo, ls="--", color="purple", lw=1.1,
               label="4.2-inch pad window", zorder=0)
    ax.axhline(pad_hi, ls="--", color="purple", lw=1.1, zorder=0)

    for f, col in ((est.left, "tab:blue"), (est.right, "tab:cyan")):
        if f.anchor >= 0:
            ax.plot(*b.pts[f.anchor], "o", color="red", ms=6, zorder=6)
        if len(f.contact_points) >= 2:
            ax.plot(
                f.contact_points[:, 0], f.contact_points[:, 1],
                color="limegreen", lw=4, zorder=4,
            )
        elif len(f.contact_points) == 1:
            ax.plot(*f.contact_points[0], ".", color=col, ms=4, zorder=4)

    if not est.pair.antipodal:
        ax.text(
            0.5, 0.5, "INFEASIBLE\nnon-antipodal side anchors",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="crimson", fontweight="bold",
        )
    elif est.contact_floor_applied:
        ax.text(
            0.5, 0.5,
            f"MINIMUM CONTACT FLOOR APPLIED\n"
            f"geometric={est.geometric_contact_fraction:.3f}, "
            f"floor={est.minimum_contact_fraction:.3f}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="darkorange", fontweight="bold",
        )

    obj_h = float(b.pts[:, 1].max() - b.pts[:, 1].min())
    obj_w = float(b.pts[:, 0].max() - b.pts[:, 0].min())
    ax.set_aspect("equal")
    ax.set_title(
        f"{title}\n"
        f"object H x W = {obj_h:.1f} x {obj_w:.1f} mm   "
        f"contact L/R = {est.left.contact_length:.2f} / "
        f"{est.right.contact_length:.2f} mm\n"
        f"geometric fraction = {est.geometric_contact_fraction:.3f}   "
        f"reported fraction = {est.combined_contact_fraction:.3f}   "
        f"antipodal = {est.pair.antipodal}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7)

    axk.plot(b.s, b.kappa, lw=0.9, color="0.3")
    axk.axhline(est.k_max_per_mm, color="crimson", ls=":", lw=1,
                label="maximum convex curvature")
    axk.axhline(0.0, color="0.8", lw=0.8)
    for f in (est.left, est.right):
        if len(f.contact_idx):
            axk.scatter(
                b.s[f.contact_idx], b.kappa[f.contact_idx],
                s=5, color="limegreen", alpha=0.55,
            )
    axk.set_xlabel("arc length s (mm)")
    axk.set_ylabel(r"$\kappa$ (1/mm)")
    curvature_extent = float(np.percentile(np.abs(b.kappa), 99))
    limit = max(1.5 * est.k_max_per_mm, 1.2 * curvature_extent, 0.02)
    axk.set_ylim(-limit, limit)
    axk.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
