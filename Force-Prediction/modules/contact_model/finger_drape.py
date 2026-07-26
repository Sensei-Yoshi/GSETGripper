"""Reachability tests for the compliant Fin-Ray contact strip.

The reduced-order model treats ``minimum_bend_radius_mm`` as the calibrated
longitudinal bend limit of the assembled finger.  A boundary point is locally
reachable when:

1. positive (convex) curvature does not exceed ``1 / minimum_bend_radius``;
2. an exterior disk with that radius, tangent at the point, contains no other
   boundary point.  The second test rejects concave pockets that the finger
   cannot enter.

Contact traversal is intentionally handled in ``contact_area.py``.  It is a
strictly contiguous walk: once reachability or side orientation fails, the
physical contact patch ends.  There is no idealised bridge or re-landing.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .contact_geometry import Boundary


def reachability_mask(
    b: Boundary, minimum_bend_radius_mm: float
) -> np.ndarray:
    """Return the locally conformable boundary samples.

    ``minimum_bend_radius_mm`` must be positive.  Signed curvature is kept:
    only positive/convex curvature is subject to the local curvature ceiling;
    concave accessibility is handled by the exterior rolling-disk test.
    """
    if minimum_bend_radius_mm <= 0:
        raise ValueError("minimum_bend_radius_mm must be positive")

    k_max = 1.0 / minimum_bend_radius_mm
    convex_ok = b.kappa <= k_max + 1e-9
    centers = b.pts + minimum_bend_radius_mm * b.N
    tree = cKDTree(b.pts)
    hits = tree.query_ball_point(
        centers, minimum_bend_radius_mm * (1.0 - 1e-6)
    )
    disk_ok = np.fromiter((len(h) == 0 for h in hits), dtype=bool, count=len(b))
    return convex_ok & disk_ok
