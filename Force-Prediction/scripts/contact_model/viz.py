"""Module 6 - overlay figures: boundary, reachability, windows, contact
segments, wrap arcs, and the curvature profile. The overlay is the sanity
check that the model behaves before trusting the numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contact_area import ContactEstimate, FingerResult


def _contact_runs(f: FingerResult) -> list[np.ndarray]:
    runs, cur = [], []
    for k, idx in enumerate(f.window_idx):
        if f.contact[k]:
            cur.append(idx)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


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
            b.pts[bad, 0], b.pts[bad, 1], s=2, color="crimson", alpha=0.5,
            zorder=2, label="unreachable",
        )

    if est.finger is not None and est.pad_band is not None:
        pad_lo, pad_hi = est.pad_band
        for yv, style, col, lab in (
            (b.pts[:, 1].min(), "-", "0.4", "table"),
            (est.tip_height, ":", "0.4", "fingertip"),
            (pad_lo, "--", "purple", "pad band"),
            (pad_hi, "--", "purple", None),
        ):
            ax.axhline(yv, ls=style, color=col, lw=1.1, label=lab, zorder=0)
        if not est.feasible:
            ax.text(
                0.5, 0.5, "INFEASIBLE\npad band above object",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="crimson", fontweight="bold",
            )

    for f, col in ((est.left, "tab:blue"), (est.right, "tab:cyan")):
        if len(f.window_idx) == 0:
            continue
        w = b.pts[f.window_idx]
        ax.plot(w[:, 0], w[:, 1], color=col, lw=6, alpha=0.18, zorder=2)
        for run in _contact_runs(f):
            seg = b.pts[run]
            ax.plot(seg[:, 0], seg[:, 1], color="limegreen", lw=4, zorder=4)
        ax.plot(*b.pts[f.anchor], "o", color="red", ms=6, zorder=6)
        for arc in f.arcs:
            phi = arc.phi0 + arc.sweep * np.linspace(0.0, arc.max_phi, 60)
            ax.plot(
                arc.center[0] + arc.r * np.cos(phi),
                arc.center[1] + arc.r * np.sin(phi),
                ls="--", lw=1.2, color="darkorange", zorder=3,
            )

    ax.set_aspect("equal")
    ax.set_title(
        f"{title}\n"
        f"contact L/R = {est.left.contact_length:.2f} / "
        f"{est.right.contact_length:.2f} mm   "
        f"area = {est.total_area:.1f} mm$^2$   "
        f"fraction = {est.mean_fraction:.3f}   "
        f"antipodal = {est.pair.antipodal}",
        fontsize=9,
    )
    if bad.any() or est.finger is not None:
        ax.legend(loc="upper right", fontsize=7)

    axk.plot(b.s, b.kappa, lw=0.9, color="0.3")
    axk.axhline(est.k_max, color="crimson", ls=":", lw=1)
    axk.axhline(0.0, color="0.8", lw=0.8)
    for f in (est.left, est.right):
        if len(f.window_idx) == 0:
            continue
        axk.axvspan(
            b.s[f.window_idx[0]], b.s[f.window_idx[-1]],
            color="tab:blue", alpha=0.12,
        )
    axk.set_xlabel("arc length s (mm)")
    axk.set_ylabel(r"$\kappa$ (1/mm)")
    lim = max(2.5 * est.k_max, 0.05)
    axk.set_ylim(-lim, lim)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
