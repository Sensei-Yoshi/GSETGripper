"""Projected contact-fraction capture tab."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from modules.contact_model import ContactParams, analyze_image, create_rembg_session
from streamlit_app.context import AppContext

# --------------------------------------------------------------------------- #
# Contact Fraction tab: fixed modeling assumptions.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Constant pad width cancels from the projected two-pad contact fraction. The
# jaws close along the image x-axis, a fixed property of the camera mount.
CONTACT_CLOSING_AXIS = "x"


@st.cache_resource(show_spinner="Loading background-removal model...")
def _rembg_session():
    """Load the rembg model once per Streamlit process."""
    return create_rembg_session()


@st.cache_resource(show_spinner="Opening camera...")
def _video_capture(index: int, width: int, height: int):
    """Open the camera once and keep it open across reruns.

    The Orbbec's RGB stream enumerates as a standard USB (UVC) video device,
    so it is opened with cv2.VideoCapture exactly like scripts/collect_images.py
    (pyorbbecsdk is only needed for depth, which this pipeline does not use).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from collect_images import open_camera

    return open_camera(index, width, height)


def _read_camera_frame(index: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """Grab one BGR frame from the cached capture device."""
    cap = _video_capture(int(index), width, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"camera index {index} returned no frame")
    return frame


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return slug or "object"


def render(context: AppContext) -> None:
    cfg = context.config
    objects_root = context.dataset.paths.objects
    st.subheader("Projected Contact-Fraction Capture")
    st.caption(
        "Capture a real object from the camera (the Orbbec's RGB stream shows "
        "up as a normal USB webcam), extract its outline, and estimate the "
        "finger-contact fraction. Each capture is saved under "
        f"`data/{context.dataset.dataset_id}/objects/<name>/` with derived "
        "contact-model outputs under `contact_fraction/`. "
        "If the wrong feed appears, change the camera index below."
    )

    p = st.columns(3)
    px_per_mm = p[0].number_input(
        "px per mm (required)", min_value=0.0,
        value=float(cfg.geometry.px_per_mm), step=0.01, format="%.4f",
        help="Camera scale. Default comes from geometry.px_per_mm in "
             "config.yaml; re-measure with scripts/calibrate_scale.py.",
    )
    minimum_bend_radius_mm = p[1].number_input(
        "minimum bend radius (mm)", min_value=1.0,
        value=float(cfg.geometry.minimum_bend_radius_mm), step=1.0,
        help="Smallest longitudinal radius the assembled Fin-Ray finger is "
             "allowed to follow. Larger values are more conservative.",
    )
    side_angle_deg = p[2].number_input(
        "side-normal tolerance (deg)", min_value=1.0, max_value=89.0,
        value=float(cfg.geometry.side_angle_deg), step=1.0,
        help="A surface counts only when its outward normal is this close to "
             "the horizontal jaw-closing direction.",
    )
    p[0].caption("default from config.yaml (geometry.px_per_mm)")

    with st.expander("Advanced parameters"):
        sweep_str = st.text_input(
            "bend-radius sweep (mm, logged)", value="10,20,30"
        )
        st.caption(
            f"Pad length is fixed to L={cfg.geometry.pad_length_mm:.2f} mm "
            "(4.2 inches) from config.yaml. The pad top is aligned to the "
            "detected object top. Constant pad width cancels from the fraction. "
            f"Valid antipodal grasps use a minimum contact fraction of "
            f"{cfg.geometry.minimum_contact_fraction:.3f}."
        )

    st.divider()
    name_in = st.text_input("Object name", placeholder="e.g. water_bottle")
    controls = st.columns([1, 1, 1])
    cam_index = int(controls[0].number_input(
        "camera index", min_value=0, value=0, step=1,
        help="Which USB video device to use. The Orbbec RGB feed is often 0 "
             "or 1; change it if the wrong camera appears."))
    live = controls[1].toggle("Live preview", value=False)
    capture = controls[2].button("Capture & Analyze", type="primary")

    if live:
        @st.fragment(run_every=0.7)
        def _preview() -> None:
            try:
                frame = _read_camera_frame(cam_index)
            except Exception as exc:  # device busy or wrong index
                st.warning(f"Camera preview unavailable: {exc}")
                return
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                use_container_width=True, caption="Live preview",
            )

        _preview()

    if capture:
        if not name_in.strip():
            st.warning("Enter an object name.")
            st.stop()
        if px_per_mm <= 0:
            st.warning("Set a real px-per-mm scale.")
            st.stop()

        name = _slugify(name_in)
        object_dir = objects_root / name
        run_dir = object_dir / "contact_fraction"
        if object_dir.exists():
            st.error(
                f"Object `{name}` already exists. Choose a new object name; "
                "existing images and analyses are preserved."
            )
            st.stop()

        try:
            frame = _read_camera_frame(cam_index)
        except Exception as exc:
            st.error("Could not read a frame from the camera. Check the USB "
                     "connection, that no other process holds it, and that the "
                     "camera index is correct.")
            st.exception(exc)
            st.stop()

        params = ContactParams(
            px_per_mm=px_per_mm,
            closing_axis=CONTACT_CLOSING_AXIS,
            pad_length_mm=float(cfg.geometry.pad_length_mm),
            minimum_bend_radius_mm=minimum_bend_radius_mm,
            side_angle_deg=side_angle_deg,
            minimum_contact_fraction=float(
                cfg.geometry.minimum_contact_fraction
            ),
            sweep_radii_mm=tuple(
                float(v) for v in sweep_str.split(",") if v.strip()
            ),
        )
        try:
            objects_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{name}-", dir=objects_root
            ) as temporary:
                staged_object = Path(temporary)
                staged_image = staged_object / "image.png"
                if not cv2.imwrite(str(staged_image), frame):
                    raise OSError(f"could not write {staged_image}")
                with st.spinner("Extracting outline and computing contact fraction..."):
                    est, summary, staged_paths = analyze_image(
                        staged_image,
                        staged_object / "contact_fraction",
                        name,
                        params,
                        session=_rembg_session(),
                    )
                staged_object.replace(object_dir)
                paths = {
                    key: object_dir / Path(path).relative_to(staged_object)
                    for key, path in staged_paths.items()
                }
        except Exception as exc:
            st.exception(exc)
            st.stop()

        st.session_state["contact_last"] = {
            "name": name, "run_dir": str(run_dir), "summary": summary,
            "paths": {k: str(v) for k, v in paths.items()},
            "feasible": bool(est.feasible),
        }

    res = st.session_state.get("contact_last")
    if res:
        st.divider()
        st.markdown(f"**Results — {res['name']}**")
        r = res["summary"]["results"]

        if not res["feasible"]:
            st.error(
                "No feasible antipodal side-contact patch under the current "
                "angle and bend-radius limits."
            )
        elif r["contact_floor_applied"]:
            st.info(
                "The configured minimum-contact floor was applied because "
                "the resolved continuous green-path fraction was smaller."
            )

        m = st.columns(4)
        m[0].metric(
            "Combined contact fraction",
            f"{r['combined_contact_fraction']:.3f}",
        )
        m[1].metric(
            "Geometric contact length",
            f"{r['combined_contact_length_mm']:.1f} mm",
        )
        m[2].metric("Contact L / R", f"{r['left']['contact_length_mm']:.1f} / "
                    f"{r['right']['contact_length_mm']:.1f} mm")
        m[3].metric("Antipodal grasp", "yes" if r["antipodal_grasp"] else "no")

        d = st.columns(4)
        d[0].metric("Object height", f"{r['object_height_mm']:.1f} mm")
        d[1].metric("Object width", f"{r['object_width_mm']:.1f} mm")
        d[2].metric(
            "Per-pad fraction L / R",
            f"{r['left']['contact_fraction']:.3f} / "
            f"{r['right']['contact_fraction']:.3f}",
        )
        d[3].metric(
            "Pad length L",
            f"{res['summary']['params']['pad_length_mm']:.1f} mm",
        )

        cols = st.columns(2)
        cols[0].image(res["paths"]["contact_fig"],
                      caption="Contact model (numbers at top)",
                      use_container_width=True)
        cols[1].image(res["paths"]["spline_overlay"],
                      caption="Fitted outline over the capture",
                      use_container_width=True)

        st.caption(
            f"Geometric fraction: {r['geometric_contact_fraction']:.4f}; "
            f"configured minimum: "
            f"{res['summary']['params']['minimum_contact_fraction']:.4f}. "
            "Minimum-bend-radius sweep (combined fraction): "
            f"{res['summary']['bend_radius_sweep_combined_fraction']}"
        )
        with st.expander("summary.json"):
            st.json(res["summary"])
        st.success(f"Saved to `{res['run_dir']}`")
