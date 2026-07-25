"""Streamlit page: capture an object from the live camera and run the
geometric contact-area model on it.

Flow: live camera feed -> "Take Photo" -> enter the object name -> "Save &
Analyze". Results land in data/test_contact_area/<object_name>/ with the
image named <object_name>.png alongside every contact-model artifact
(spline overlay, CSV, SVG, cutout, mask, contact figure, summary.json), and
a row is appended to data/test_contact_area/index.csv.

Run from the repo root:  streamlit run app.py   (this appears in the sidebar)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "contact_model"))

from contact_area import FingerGeometry  # noqa: E402
from pipeline_core import ContactParams, analyze_image  # noqa: E402

TEST_ROOT = REPO / "data" / "test_contact_area"
INDEX_CSV = TEST_ROOT / "index.csv"

st.set_page_config(page_title="Contact Area Test", page_icon="CA", layout="wide")


@st.cache_resource(show_spinner="Loading background-removal model...")
def get_rembg_session():
    """Cache the rembg session so it loads once per Streamlit process."""
    from rembg import new_session

    import extract_object_outline as outline

    return new_session(outline.REMBG_MODEL)


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return slug or "object"


def decode_photo(buf) -> np.ndarray | None:
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR, matches cv2.imread


st.title("Contact-Area Test Capture")
st.caption(
    "Capture a real object, extract its outline, and estimate the "
    "finger-contact fraction. Saved under `data/test_contact_area/<name>/`."
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model parameters")
    px_per_mm = st.number_input(
        "px per mm (REQUIRED)", min_value=0.0, value=8.0, step=0.1,
        help="From a fiducial of known width in the scene. Every mm-valued "
             "parameter is meaningless without a real scale.",
    )
    object_type = st.selectbox(
        "Object type", ["axisymmetric", "prismatic"],
        help="axisymmetric: bottles/cans/fruit (transverse width from the "
             "silhouette). prismatic: flat-faced (full pad width).",
    )
    closing_axis = st.selectbox("Jaw closing axis in image", ["x", "y"])
    c1, c2 = st.columns(2)
    k_max = c1.number_input("k_max (1/mm)", min_value=0.01, value=2.0, step=0.5)
    delta = c2.number_input("delta (mm)", min_value=0.0, value=0.3, step=0.1)
    L = c1.number_input("pad length L (mm)", min_value=0.5, value=4.0, step=0.5)
    w_pad = c2.number_input("pad width (mm)", min_value=0.5, value=12.0, step=1.0)
    sweep_str = st.text_input("k_max sweep (logged)", value="1,2,4")

    st.divider()
    use_finger = st.checkbox(
        "Finger drop-depth model", value=False,
        help="Constrain the pad's height band by finger geometry instead of "
             "free placement. Needed for short objects (fruit) where the pad "
             "cannot reach the equator.",
    )
    finger = None
    if use_finger:
        finger_length = st.number_input(
            "finger length: palm to tip (mm)", min_value=1.0, value=100.0)
        pad_start = st.number_input(
            "pad start: tip to pad edge (mm)", min_value=0.0, value=0.0)
        tip_clearance = st.number_input(
            "tip clearance above table (mm)", min_value=0.0, value=2.0)
        palm_standoff = st.number_input(
            "palm standoff above object (mm)", min_value=0.0, value=5.0)
        finger = FingerGeometry(
            finger_length=finger_length, pad_length=L, pad_start=pad_start,
            tip_clearance=tip_clearance, palm_standoff=palm_standoff,
        )

overwrite = st.checkbox(
    "Overwrite if a folder with this name already exists", value=False)

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
st.subheader("1. Capture")
photo = st.camera_input(
    "Live camera - frame the object alone on a plain background, then Take Photo")

st.subheader("2. Name and analyze")
name_in = st.text_input("Object name", placeholder="e.g. water_bottle")
go = st.button("Save & Analyze", type="primary", disabled=photo is None)

if go:
    if photo is None:
        st.warning("Take a photo first.")
        st.stop()
    if not name_in.strip():
        st.warning("Enter an object name.")
        st.stop()
    if px_per_mm <= 0:
        st.warning("Set a real px-per-mm scale in the sidebar.")
        st.stop()

    name = slugify(name_in)
    run_dir = TEST_ROOT / name
    if run_dir.exists() and not overwrite:
        st.error(
            f"`{run_dir}` already exists. Enable 'Overwrite' or pick another "
            "name.")
        st.stop()

    frame = decode_photo(photo)
    if frame is None:
        st.error("Could not decode the captured photo.")
        st.stop()

    run_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_dir / f"{name}.png"
    cv2.imwrite(str(image_path), frame)

    params = ContactParams(
        px_per_mm=px_per_mm, object_type=object_type, closing_axis=closing_axis,
        k_max=k_max, delta=delta, L=L, w_pad=w_pad,
        sweep_k=tuple(float(v) for v in sweep_str.split(",") if v.strip()),
        finger=finger,
    )

    try:
        with st.spinner("Extracting outline and computing contact area..."):
            est, summary, paths = analyze_image(
                image_path, run_dir, name, params,
                session=get_rembg_session(), index_csv=INDEX_CSV,
            )
    except Exception as exc:  # surface extractor/model failures in the UI
        st.exception(exc)
        st.stop()

    st.session_state["last_result"] = {
        "name": name, "run_dir": str(run_dir), "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "feasible": bool(est.feasible),
    }

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.get("last_result")
if res:
    st.divider()
    st.subheader(f"3. Results - {res['name']}")
    r = res["summary"]["results"]

    if res["summary"].get("finger") and not res["feasible"]:
        st.error("Grasp INFEASIBLE: the pad's height band sits above the "
                 "object top. Reported contact is zero.")

    m = st.columns(4)
    m[0].metric("Mean contact fraction", f"{r['mean_fraction']:.3f}")
    m[1].metric("Total area", f"{r['total_area_mm2']:.1f} mm2")
    m[2].metric("Contact L / R",
                f"{r['left']['contact_mm']:.1f} / {r['right']['contact_mm']:.1f} mm")
    m[3].metric("Antipodal grasp", "yes" if r["antipodal_grasp"] else "no")

    left, right = st.columns(2)
    with left:
        st.image(res["paths"]["contact_fig"],
                 caption="Contact model (numbers at top)", use_container_width=True)
    with right:
        st.image(res["paths"]["spline_overlay"],
                 caption="Fitted outline over the capture", use_container_width=True)

    st.caption(f"k_max sweep (mean fraction): "
               f"{res['summary']['k_max_sweep_mean_fraction']}")
    with st.expander("summary.json"):
        st.json(res["summary"])
    st.success(f"Saved to `{res['run_dir']}`")
