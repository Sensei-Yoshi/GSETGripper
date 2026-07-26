"""Standalone analytic smoke suite for the contact-fraction estimator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contact_area import estimate_contact
from .synthetic_shapes import circle, rounded_rect
from .viz import plot_estimate

OUT = Path(__file__).resolve().parent / "test_outputs"
PAD_LENGTH_MM = 106.68
MINIMUM_BEND_RADIUS_MM = 20.0
SIDE_ANGLE_DEG = 30.0
MINIMUM_CONTACT_FRACTION = 0.05


def estimate(points: np.ndarray, ds: float = 0.25):
    return estimate_contact(
        points,
        pad_length_mm=PAD_LENGTH_MM,
        minimum_bend_radius_mm=MINIMUM_BEND_RADIUS_MM,
        side_angle_deg=SIDE_ANGLE_DEG,
        minimum_contact_fraction=MINIMUM_CONTACT_FRACTION,
        ds=ds,
        smoothing_mm=0.0,
    )


def main() -> int:
    tall = rounded_rect(60.0, 220.0, 0.1) + np.array([0.0, 110.0])
    short = rounded_rect(60.0, 60.0, 0.1) + np.array([0.0, 30.0])
    gentle = circle(40.0) + np.array([0.0, 40.0])
    tight = circle(8.0) + np.array([0.0, 8.0])
    cases = {
        "v2_tall_rectangle": tall,
        "v2_short_rectangle": short,
        "v2_gentle_circle_R40": gentle,
        "v2_tight_circle_R8": tight,
    }

    estimates = {name: estimate(points) for name, points in cases.items()}
    for name, result in estimates.items():
        plot_estimate(result, OUT / f"{name}.png", name)

    expected_circle = np.pi * 40.0 / 3.0
    checks = {
        "tall rectangle near full": estimates["v2_tall_rectangle"].combined_contact_fraction > 0.98,
        "short rectangle scales with height": abs(
            estimates["v2_short_rectangle"].combined_contact_fraction
            - 60.0 / PAD_LENGTH_MM
        ) < 0.02,
        "gentle circle side-angle truth": abs(
            estimates["v2_gentle_circle_R40"].left.contact_length
            - expected_circle
        ) < 0.8,
        "tight circle uses contact floor": (
            estimates["v2_tight_circle_R8"].combined_contact_fraction
            == MINIMUM_CONTACT_FRACTION
            and estimates["v2_tight_circle_R8"].contact_floor_applied
        ),
    }

    failures = 0
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
        failures += 0 if passed else 1
    print(f"figures: {OUT}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
