from __future__ import annotations

import csv
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from modules.contact_model import ContactParams, build_summary, estimate_contact
from modules.contact_model.contact_area import _rigid_pad_top_candidates
from modules.contact_model.synthetic_shapes import circle, rounded_rect

ROOT = Path(__file__).resolve().parents[1]
CONTACT_MODEL = ROOT / "modules" / "contact_model"

PAD_LENGTH_MM = 106.68
FINGER_EXTENSION_MM = 63.5
RADIUS_MM = 20.0
SIDE_ANGLE_DEG = 30.0
MINIMUM_CONTACT_FRACTION = 0.05
PLANAR_THRESHOLD = 0.15


def _estimate(points: np.ndarray, *, ds: float = 0.1, radius: float = RADIUS_MM):
    return estimate_contact(
        points,
        pad_length_mm=PAD_LENGTH_MM,
        minimum_bend_radius_mm=radius,
        side_angle_deg=SIDE_ANGLE_DEG,
        minimum_contact_fraction=MINIMUM_CONTACT_FRACTION,
        ds=ds,
        smoothing_mm=0.0,
    )


def _estimate_rigid(points: np.ndarray, *, ds: float = 0.1, tol: float = 0.5):
    return estimate_contact(
        points,
        mode="rigid",
        pad_length_mm=PAD_LENGTH_MM,
        side_angle_deg=SIDE_ANGLE_DEG,
        minimum_contact_fraction=MINIMUM_CONTACT_FRACTION,
        rigid_contact_tolerance_mm=tol,
        finger_extension_mm=FINGER_EXTENSION_MM,
        planar_threshold=PLANAR_THRESHOLD,
        ds=ds,
        smoothing_mm=0.0,
    )


def _rounded_box(width: float, height: float) -> np.ndarray:
    return rounded_rect(width, height, 0.1) + np.array([0.0, height / 2.0])


def _line(start, end, count: int = 600) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    fraction = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
    return start + fraction * (end - start)


def _load_fixture_outline(name: str) -> np.ndarray:
    csv_path = CONTACT_MODEL / "test_contact_area" / name / f"{name}_spline_points.csv"
    with csv_path.open() as file:
        reader = csv.reader(file)
        next(reader)
        points = np.array([[float(x), float(y)] for x, y in reader])
    points /= 2.1852
    points[:, 1] = points[:, 1].max() - points[:, 1]
    return points


def _assert_all_green_points_are_side_facing(estimate) -> None:
    threshold = np.cos(np.radians(SIDE_ANGLE_DEG))
    for finger in (estimate.left, estimate.right):
        if len(finger.contact_idx) == 0:
            continue
        side_sign = -1.0 if finger.side == "left" else 1.0
        alignment = side_sign * estimate.boundary.N[finger.contact_idx, 0]
        assert np.all(alignment >= threshold - 1e-9)


def test_tall_rectangle_is_nearly_full_without_top_or_bottom_contact():
    estimate = _estimate(_rounded_box(60.0, 220.0))
    assert estimate.pair.antipodal
    assert estimate.combined_contact_fraction == pytest.approx(1.0, abs=0.01)
    assert estimate.left.contact_length <= PAD_LENGTH_MM
    assert estimate.right.contact_length <= PAD_LENGTH_MM
    _assert_all_green_points_are_side_facing(estimate)


def test_short_and_thin_rectangles_scale_with_available_side_height():
    short = _estimate(_rounded_box(60.0, 60.0))
    thin = _estimate(_rounded_box(60.0, 5.0))
    assert short.combined_contact_fraction == pytest.approx(
        60.0 / PAD_LENGTH_MM, abs=0.02
    )
    assert thin.geometric_contact_fraction < MINIMUM_CONTACT_FRACTION
    assert thin.combined_contact_fraction == MINIMUM_CONTACT_FRACTION
    assert thin.contact_floor_applied


def test_gentle_circle_stops_at_thirty_degree_side_boundary():
    radius = 40.0
    estimate = _estimate(circle(radius) + np.array([0.0, radius]), ds=0.25)
    expected_per_pad = np.pi * radius / 3.0
    assert estimate.left.contact_length == pytest.approx(expected_per_pad, abs=0.8)
    assert estimate.right.contact_length == pytest.approx(expected_per_pad, abs=0.8)
    _assert_all_green_points_are_side_facing(estimate)


def test_tight_circle_uses_minimum_contact_floor():
    estimate = _estimate(circle(8.0) + np.array([0.0, 8.0]))
    assert estimate.geometric_contact_fraction == 0.0
    assert estimate.combined_contact_fraction == MINIMUM_CONTACT_FRACTION
    assert estimate.contact_floor_applied
    assert estimate.feasible


def test_waist_stops_at_first_failure_without_relanding():
    waist = rounded_rect(50.0, 90.0, 8.0, notch_r=5.0) + np.array([0.0, 45.0])
    estimate = _estimate(waist, ds=0.25)
    threshold = np.cos(np.radians(SIDE_ANGLE_DEG))
    pad_lo, pad_hi = estimate.pad_band
    for finger in (estimate.left, estimate.right):
        assert len(finger.contact_idx) >= 2
        steps = np.diff(finger.contact_idx) % len(estimate.boundary)
        assert np.all(steps == 1)
        side_sign = -1.0 if finger.side == "left" else 1.0
        y = estimate.boundary.pts[:, 1]
        locally_valid = (
            estimate.reachable
            & (side_sign * estimate.boundary.N[:, 0] >= threshold)
            & (y >= pad_lo)
            & (y <= pad_hi)
        )
        # The notched outline contains a later valid side region. A bridging
        # model would re-land there; the contiguous model intentionally does not.
        assert np.any(
            locally_valid
            & ~np.isin(np.arange(len(estimate.boundary)), finger.contact_idx)
        )


def test_non_antipodal_shape_is_infeasible_with_zero_primary_fraction():
    height = 60.0
    top_width = 40.0
    bottom_width = 2.0 * (
        top_width / 2.0 + height * np.tan(np.radians(SIDE_ANGLE_DEG))
    )
    points = np.vstack([
        _line((bottom_width / 2.0, 0.0), (top_width / 2.0, height)),
        _line((top_width / 2.0, height), (-top_width / 2.0, height)),
        _line((-top_width / 2.0, height), (-bottom_width / 2.0, 0.0)),
        _line((-bottom_width / 2.0, 0.0), (bottom_width / 2.0, 0.0)),
    ])
    estimate = _estimate(points)
    assert not estimate.pair.antipodal
    assert not estimate.feasible
    assert not estimate.contact_floor_applied
    assert estimate.combined_contact_fraction == 0.0


def test_numerical_invariants_and_resampling_stability():
    points = circle(40.0) + np.array([0.0, 40.0])
    estimates = [_estimate(points, ds=ds) for ds in (0.1, 0.25, 0.5)]
    fractions = [estimate.combined_contact_fraction for estimate in estimates]
    assert max(fractions) - min(fractions) < 0.01
    for estimate in estimates:
        assert 0.0 <= estimate.left.contact_length <= PAD_LENGTH_MM
        assert 0.0 <= estimate.right.contact_length <= PAD_LENGTH_MM
        assert estimate.combined_contact_length <= 2.0 * PAD_LENGTH_MM
        assert 0.0 <= estimate.geometric_contact_fraction <= 1.0
        assert 0.0 <= estimate.combined_contact_fraction <= 1.0


@pytest.mark.parametrize(
    ("name", "legacy_fraction", "maximum_new_fraction"),
    [("water_bottle", 1.0, 0.95), ("test_3d_print", 0.823, 0.60)],
)
def test_saved_capture_regressions_are_side_only_and_radius_monotone(
    name: str, legacy_fraction: float, maximum_new_fraction: float
):
    points = _load_fixture_outline(name)
    estimates = [
        estimate_contact(
            points,
            pad_length_mm=PAD_LENGTH_MM,
            minimum_bend_radius_mm=radius,
            side_angle_deg=SIDE_ANGLE_DEG,
            minimum_contact_fraction=MINIMUM_CONTACT_FRACTION,
            ds=0.25,
            smoothing_mm=0.2,
        )
        for radius in (10.0, 20.0, 30.0)
    ]
    fractions = [estimate.combined_contact_fraction for estimate in estimates]
    assert fractions[0] >= fractions[1] >= fractions[2]
    assert fractions[1] < legacy_fraction
    assert fractions[1] < maximum_new_fraction
    for estimate in estimates:
        _assert_all_green_points_are_side_facing(estimate)


def test_contact_api_has_no_width_or_absolute_area_parameter():
    parameters = inspect.signature(estimate_contact).parameters
    assert "w_pad" not in parameters
    assert "object_type" not in parameters
    assert parameters["minimum_contact_fraction"].default == 0.05
    estimate = _estimate(circle(40.0) + np.array([0.0, 40.0]))
    assert not hasattr(estimate, "total_area")


def test_v2_summary_excludes_pad_width_and_area():
    estimate = _estimate(circle(40.0) + np.array([0.0, 40.0]))
    params = ContactParams(px_per_mm=2.1852)
    summary = build_summary(
        "analytic_circle",
        "analytic_circle.png",
        estimate,
        params,
        {"10": 0.5, "20": 0.4, "30": 0.3},
    )
    serialized = repr(summary).lower()
    assert summary["schema_version"] == 2
    assert summary["metric"] == "projected_two_pad_contact_fraction"
    assert summary["params"]["minimum_contact_fraction"] == 0.05
    assert "geometric_contact_fraction" in summary["results"]
    assert "contact_floor_applied" in summary["results"]
    assert "w_pad" not in serialized
    assert "pad_width" not in serialized
    assert "area_mm" not in serialized

def test_rigid_tall_planar_box_is_full_contact_and_planar():
    estimate = _estimate_rigid(_rounded_box(60.0, 220.0))
    assert estimate.mode == "rigid"
    assert estimate.is_planar
    assert estimate.pair.antipodal
    assert estimate.combined_contact_fraction == pytest.approx(1.0, abs=0.01)
    assert estimate.left.contact_length <= PAD_LENGTH_MM
    assert estimate.right.contact_length <= PAD_LENGTH_MM
    _assert_all_green_points_are_side_facing(estimate)


def test_rigid_cylinder_is_non_planar_with_small_contact():
    estimate = _estimate_rigid(circle(40.0) + np.array([0.0, 40.0]), ds=0.25)
    assert estimate.mode == "rigid"
    assert not estimate.is_planar
    assert estimate.geometric_contact_fraction < PLANAR_THRESHOLD
    assert estimate.feasible
    assert estimate.combined_contact_fraction == max(
        estimate.geometric_contact_fraction, MINIMUM_CONTACT_FRACTION
    )


def test_rigid_tight_cylinder_falls_to_minimum_contact_floor():
    estimate = _estimate_rigid(circle(6.0) + np.array([0.0, 6.0]))
    assert not estimate.is_planar
    assert estimate.geometric_contact_fraction < MINIMUM_CONTACT_FRACTION
    assert estimate.combined_contact_fraction == MINIMUM_CONTACT_FRACTION
    assert estimate.contact_floor_applied


def test_rigid_placement_slides_down_to_a_lower_straight_face():
    # A shape with straight vertical faces only in its lower band and a spike
    # that tapers to a point above; the rigid window should slide down onto the
    # straight faces instead of staying aligned to the (non-side-facing) top.
    width, straight_top, apex = 30.0, 120.0, 260.0
    lower = np.linspace(0.0, straight_top, 300)
    upper = np.linspace(straight_top, apex, 300)

    def taper(y):
        return width * (1.0 - (y - straight_top) / (apex - straight_top))

    right = (
        [(width, y) for y in lower]
        + [(taper(y), y) for y in upper]
    )
    left = (
        [(-taper(y), y) for y in reversed(upper)]
        + [(-width, y) for y in reversed(lower)]
    )
    points = np.array(right + [(0.0, apex)] + left + [(-width, 0.0), (width, 0.0)])
    estimate = _estimate_rigid(points, ds=0.25)
    assert estimate.is_planar
    # The chosen window top is pushed well below the object top toward the
    # straight region, and cannot exceed the 63.5 mm downward-travel cap.
    object_top = float(estimate.boundary.pts[:, 1].max())
    assert object_top - estimate.pad_band[1] > 10.0
    assert estimate.pad_band[1] >= object_top - FINGER_EXTENSION_MM - 1e-6
    assert estimate.left.contact_length > 0.0
    assert estimate.right.contact_length > 0.0


def test_rigid_pad_top_candidates_respect_downward_travel():
    # Tall object: full 2.5 in travel.
    tall = _rigid_pad_top_candidates(300.0, 0.0, PAD_LENGTH_MM, FINGER_EXTENSION_MM, 0.25)
    assert max(tall) == pytest.approx(300.0)
    assert min(tall) == pytest.approx(300.0 - FINGER_EXTENSION_MM)
    # Between one and one-and-a-half pad lengths: travel is height - pad length,
    # so the lowest pad top seats the pad bottom on the object bottom.
    mid = _rigid_pad_top_candidates(140.0, 0.0, PAD_LENGTH_MM, FINGER_EXTENSION_MM, 0.25)
    assert min(mid) == pytest.approx(PAD_LENGTH_MM)
    # Object shorter than the pad: no vertical freedom.
    short = _rigid_pad_top_candidates(60.0, 0.0, PAD_LENGTH_MM, FINGER_EXTENSION_MM, 0.25)
    assert short == [60.0]


def test_rigid_summary_records_mode_planarity_and_omits_bend_radius():
    estimate = _estimate_rigid(_rounded_box(60.0, 220.0))
    params = ContactParams(px_per_mm=2.1852, mode="rigid")
    summary = build_summary("planar_box", "planar_box.png", estimate, params, {})
    params_out = summary["params"]
    assert params_out["mode"] == "rigid"
    assert params_out["placement"] == "maximized_contact_area"
    assert "minimum_bend_radius_mm" not in params_out
    assert "k_max_per_mm" not in params_out
    assert params_out["finger_extension_mm"] == FINGER_EXTENSION_MM
    assert summary["results"]["is_planar"] is True
    assert "chosen_pad_top_mm" in summary["results"]
    assert summary["bend_radius_sweep_combined_fraction"] == {}


@pytest.mark.skipif(
    os.environ.get("RUN_CONTACT_IMAGE_INTEGRATION") != "1",
    reason="set RUN_CONTACT_IMAGE_INTEGRATION=1 when the local rembg model is available",
)
def test_saved_raw_image_end_to_end(tmp_path: Path):
    from modules.contact_model import analyze_image

    image = CONTACT_MODEL / "test_contact_area" / "water_bottle" / "water_bottle.png"
    params = ContactParams(px_per_mm=2.1852)
    estimate, summary, paths = analyze_image(
        image, tmp_path, "water_bottle", params
    )
    assert 0.0 <= estimate.combined_contact_fraction <= 1.0
    assert summary["schema_version"] == 2
    assert paths["contact_fig"].exists()
