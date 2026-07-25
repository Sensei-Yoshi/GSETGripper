"""Streamlit research viewer for the synthetic two-gripper Exp-Force pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from force_prediction.config import EXPERIMENT_IDS, Config, load_config
from force_prediction.contracts import Gripper, group_by_object
from force_prediction.experiments import EXPERIMENT_CATALOG, experiment_display_name
from force_prediction.expforce import (
    PREPARATION_RELATIVE,
    RESULTS_RELATIVE,
    load_experience_pool,
    load_image,
    load_prepared_descriptors,
    load_rows,
    load_saved_runs,
    pipeline_result_from_dict,
    prepare_dataset,
    run_benchmark,
    save_benchmark,
    save_pipeline_run,
    source_sha256,
    validation_summary,
)
from force_prediction.pipeline import Pipeline, PipelineRunResult, QueryInput
from force_prediction.retrieval import normalized_weights

# The geometric contact-area model lives outside the package, under scripts/;
# add it to the path so the Contact-Area tab's lazy imports resolve.
_CONTACT_MODEL_DIR = Path(__file__).resolve().parent / "scripts" / "contact_model"
if str(_CONTACT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTACT_MODEL_DIR))

st.set_page_config(page_title="Force Pipeline Lab", page_icon="FP", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink:#17211b; --muted:#66736b; --line:#dce4df; --accent:#147a4a; --warm:#b45f19; }
    .stApp { background:#f7faf8; color:var(--ink); }
    .block-container { padding-top:1.4rem; max-width:1500px; }
    h1, h2, h3 { letter-spacing:0 !important; }
    h1 { font-size:2rem !important; }
    h2 { font-size:1.35rem !important; }
    [data-testid="stMetric"] { border-top:2px solid var(--line); padding-top:.7rem; }
    [data-testid="stMetricValue"] { font-size:1.55rem; }
    .synthetic-note { border-left:4px solid var(--warm); padding:.55rem .8rem; background:#fff9f3; }
    .formula { border-left:4px solid var(--accent); padding:.45rem .8rem; background:#f0f7f3; }
    .status-ok { color:var(--accent); font-weight:650; }
    .status-warn { color:var(--warm); font-weight:650; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); }
    button { border-radius:6px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_static() -> tuple[list, dict]:
    cfg = load_config().model_copy(deep=True)
    rows = load_rows(cfg)
    return rows, validation_summary(cfg, rows)


# --------------------------------------------------------------------------- #
# Contact-Area tab: fixed modeling assumptions.
# --------------------------------------------------------------------------- #
TEST_CONTACT_ROOT = Path(__file__).resolve().parent / "data" / "test_contact_area"
# The pad is assumed to contact its full width on every object (no out-of-plane
# width reduction), so contact fraction = contact length / pad length. This is
# the "prismatic" transverse model (w_eff = w_pad). The jaws close along the
# image x-axis, a fixed property of the camera/gripper mount.
CONTACT_OBJECT_TYPE = "prismatic"
CONTACT_CLOSING_AXIS = "x"


@st.cache_resource(show_spinner="Loading background-removal model...")
def _rembg_session():
    """Load the rembg model once per Streamlit process."""
    import extract_object_outline as outline_mod
    from rembg import new_session

    return new_session(outline_mod.REMBG_MODEL)


@st.cache_resource(show_spinner="Opening camera...")
def _video_capture(index: int, width: int, height: int):
    """Open the camera once and keep it open across reruns.

    The Orbbec's RGB stream enumerates as a standard USB (UVC) video device,
    so it is opened with cv2.VideoCapture exactly like scripts/collect_images.py
    (pyorbbecsdk is only needed for depth, which this pipeline does not use).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
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


def _run_config(
    base: Config,
    *,
    live: bool,
    use_projected_contact: bool,
    semantic: float,
    mass: float,
    roughness: float,
    contact: float,
    sigma_mass: float,
    sigma_contact: float,
) -> Config:
    cfg = base.model_copy(deep=True)
    cfg.models.dry_run = not live
    cfg.inputs.use_projected_contact = use_projected_contact
    cfg.retrieval.weights.semantic = semantic
    cfg.retrieval.weights.mass = mass
    cfg.retrieval.weights.roughness = roughness
    cfg.retrieval.weights.contact = contact
    cfg.retrieval.sigma_mass = sigma_mass
    cfg.retrieval.sigma_contact = sigma_contact
    normalized_weights(cfg)
    return cfg


def _decode_upload(uploaded) -> np.ndarray | None:  # noqa: ANN001
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _format_force(value: float, *, signed: bool = False) -> str:
    spec = "+.6g" if signed else ".6g"
    return f"{format(value, spec)} N"


def _format_experiment(value: str) -> str:
    return experiment_display_name(value) if value.lower() in EXPERIMENT_CATALOG else value


@st.cache_data(show_spinner=False)
def _thumbnail(path: str, modified_ns: int, max_width: int = 420) -> np.ndarray | None:
    del modified_ns
    image = cv2.imread(path)
    if image is None:
        return None
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_width, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return _rgb(image)


def _truth_for_display(obj) -> dict:  # noqa: ANN001
    optimal, _ = obj.optimal_grippers()
    if len(optimal) != 1:
        raise ValueError(f"validation object {obj.object_id!r} does not have a strict winner")
    return {
        "true_gecko_force_n": obj.gecko.min_force_n if obj.gecko else None,
        "true_silicone_force_n": obj.silicone.min_force_n if obj.silicone else None,
        "true_selection": next(iter(optimal)).value,
    }


def _truth_payload(obj) -> dict:  # noqa: ANN001
    return {
        **_truth_for_display(obj),
        "object_id": obj.object_id,
        "gecko_feasible": obj.gecko.feasible if obj.gecko else None,
        "silicone_feasible": obj.silicone.feasible if obj.silicone else None,
    }


def _paired_retrieval_table(result: PipelineRunResult) -> pd.DataFrame:
    rows = []
    for item in result.retrieved_objects:
        sim = item.similarity
        rows.append(
            {
                "rank": item.rank,
                "object": item.object_id.replace("_", " "),
                "score": item.score,
                "semantic": sim.semantic,
                "mass": sim.mass,
                "roughness": sim.roughness,
                "contact": sim.contact,
                "mass_g": item.mass_g,
                "roughness_class": item.roughness_class,
                "contact_fraction": item.projected_contact_fraction,
                "gecko_force_n": item.gecko_min_force_n,
                "gecko_feasible": item.gecko_feasible,
                "silicone_force_n": item.silicone_min_force_n,
                "silicone_feasible": item.silicone_feasible,
            }
        )
    return pd.DataFrame(rows)


def _render_formula(cfg: Config) -> None:
    weights = normalized_weights(cfg)
    st.markdown('<div class="formula"><b>Hybrid similarity</b></div>', unsafe_allow_html=True)
    st.latex(
        r"S_i=w_s\cos(e_q,e_i)+w_m e^{-|\ln m_q-\ln m_i|/\sigma_m}"
        r"+w_r(1-|r_q-r_i|/4)+w_a e^{-|a_q-a_i|/\sigma_a}"
    )
    st.caption(
        "Normalized weights: "
        + " | ".join(f"{name} {value:.2f}" for name, value in weights.items())
    )


def _render_prediction(
    detailed: PipelineRunResult,
    truth,
    *,
    counterfactual: bool,
    baseline: PipelineRunResult | None,
    cfg: Config,
    experiment: str | None = None,
) -> None:
    result = detailed.selection
    metric_cols = st.columns(4)
    selected_label = result.desired_gripper.title()
    if result.prediction_tie:
        selected_label += " (tie-break)"
    metric_cols[0].metric("Selected gripper", selected_label)
    metric_cols[1].metric(
        "Selected force",
        _format_force(result.predicted_normal_force_n)
        if result.predicted_normal_force_n is not None
        else "None",
    )
    metric_cols[2].metric(
        "VLM recommendation",
        result.model_recommended_gripper.title()
        if result.model_recommended_gripper is not None
        else "Not applicable",
    )
    active_experiment = experiment or st.session_state.get("last_experiment", "e4")
    metric_cols[3].metric("Experiment", _format_experiment(active_experiment))

    if result.recommendation_agrees_with_selector is False:
        st.warning(
            "The VLM recommendation disagreed with the authoritative Python selector. "
            "The displayed command uses the lowest feasible predicted force."
        )

    if result.prediction_tie:
        st.warning(
            "Both grippers have the same predicted command force. Selection used "
            f"{result.tie_break_reason or 'the deterministic fallback rule'}."
        )

    if counterfactual:
        st.markdown(
            '<p class="status-warn">Counterfactual mode: source labels are not scored.</p>',
            unsafe_allow_html=True,
        )
        if baseline is not None:
            delta = (
                (result.predicted_normal_force_n or 0.0)
                - (baseline.selection.predicted_normal_force_n or 0.0)
            )
            st.metric("Selected-force change from baseline", _format_force(delta, signed=True))
    else:
        truth_values = _truth_for_display(truth)
        predicted_correct = (
            result.desired_gripper in {g.value for g in truth.optimal_grippers()[0]}
        )
        st.markdown(
            f'<p class="status-ok">Leave-one-out truth: {truth_values["true_selection"]}; '
            f'prediction {"correct" if predicted_correct else "incorrect"}.</p>',
            unsafe_allow_html=True,
        )
        ground_truth = st.columns(3)
        ground_truth[0].metric(
            "True gecko force",
            _format_force(truth_values["true_gecko_force_n"])
            if truth_values["true_gecko_force_n"] is not None
            else "Infeasible",
        )
        ground_truth[1].metric(
            "True silicone force",
            _format_force(truth_values["true_silicone_force_n"])
            if truth_values["true_silicone_force_n"] is not None
            else "Infeasible",
        )
        ground_truth[2].metric("True winning gripper", truth_values["true_selection"].title())

    pred_cols = st.columns(2)
    for column, gripper in zip(pred_cols, ("gecko", "silicone"), strict=True):
        pred = result.candidate_predictions[gripper]
        with column:
            st.subheader(gripper.title())
            truth_force = None
            if not counterfactual:
                truth_record = truth.get(Gripper(gripper))
                truth_force = truth_record.min_force_n if truth_record else None
            st.metric(
                "Predicted force",
                _format_force(pred.predicted_normal_force_n),
                delta=(
                    f"{_format_force(pred.predicted_normal_force_n - truth_force, signed=True)} error"
                    if truth_force is not None
                    else None
                ),
                delta_color="inverse",
            )
            physics = detailed.physics_estimates.get(gripper)
            if physics:
                raw_force = physics.get("raw_force_n")
                raw_label = _format_force(raw_force) if raw_force is not None else "infeasible"
                st.caption(f"Physics model: continuous estimate {raw_label}")
            else:
                st.caption("Physics model: not used by this experiment")
            st.write(pred.reasoning_trace or "No reasoning trace for this experiment.")

    st.subheader(f"Top {cfg.retrieval.k} reference matches")
    if detailed.retrieved_objects:
        if cfg.models.dry_run:
            st.caption(
                "E4 retrieves each object once with both gripper outcomes. Offline mode "
                "uses the deterministic paired-neighbor stand-in and makes no VLM request."
            )
        else:
            st.caption(
                "E4 retrieves each object once. Both gripper outcomes come from the same "
                "paired experience and are sent together in one VLM request."
            )
        st.dataframe(
            _paired_retrieval_table(detailed),
            hide_index=True,
            width="stretch",
            column_config={key: st.column_config.NumberColumn(format="%.3f") for key in (
                "score", "semantic", "mass", "roughness", "contact"
            )},
        )
    else:
        st.info("This experiment does not use retrieval.")
    _render_formula(cfg)


def single_run_view(base_cfg: Config, rows: list) -> None:
    records = load_experience_pool(base_cfg)
    objects = group_by_object(records)
    prepared = load_prepared_descriptors(base_cfg)
    names = {row.object_name: row.object_id for row in rows}

    controls, output = st.columns([0.38, 0.62], gap="large")
    with controls:
        st.subheader("Query")
        selected_name = st.selectbox("Dataset object", sorted(names))
        object_id = names[selected_name]
        truth = objects[object_id]
        sample = truth.gecko or truth.silicone
        assert sample is not None
        source_image = load_image(base_cfg, sample)
        uploaded = st.file_uploader("Override image", type=["png", "jpg", "jpeg"])
        upload_image = _decode_upload(uploaded)
        query_image = upload_image if upload_image is not None else source_image
        if query_image is not None:
            st.image(_rgb(query_image), width="stretch")
        else:
            st.warning("Image is not available locally. Prepare images before a live run.")

        mass = st.number_input("Mass (g)", min_value=0.1, value=float(sample.mass_g), step=1.0)
        roughness = st.select_slider(
            "Roughness class",
            options=[1, 2, 3, 4, 5],
            value=int(sample.roughness_class),
        )
        contact = st.slider(
            "Projected contact fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(sample.projected_contact_fraction),
            step=0.001,
        )
        experiment = st.selectbox(
            "Experiment profile",
            list(EXPERIMENT_IDS),
            index=2,
            format_func=_format_experiment,
        )
        mode = st.segmented_control("Execution", ["Offline", "Live Gemini"], default="Offline")
        live_execution = mode == "Live Gemini"

        with st.expander("Input and retrieval tuning", expanded=True):
            use_projected_contact = st.checkbox(
                "Use projected contact fraction",
                value=base_cfg.inputs.use_projected_contact,
                help=(
                    "When disabled, contact is omitted from VLM inputs and its retrieval "
                    "weight is set to zero; the remaining weights are renormalized. "
                    "Physics-based E5 and E6 still require the measured contact fraction."
                ),
            )
            semantic_w = st.slider(
                "Semantic weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.semantic), 0.05,
            )
            mass_w = st.slider(
                "Mass weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.mass), 0.05,
            )
            roughness_w = st.slider(
                "Roughness weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.roughness), 0.05,
            )
            configured_contact_w = st.slider(
                "Contact weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.contact), 0.05,
                disabled=not use_projected_contact,
            )
            contact_w = configured_contact_w if use_projected_contact else 0.0
            sigma_mass = st.slider(
                "Mass sigma", 0.1, 3.0,
                float(base_cfg.retrieval.sigma_mass), 0.1,
            )
            sigma_contact = st.slider(
                "Contact sigma", 0.05, 1.0,
                float(base_cfg.retrieval.sigma_contact), 0.05,
                disabled=not use_projected_contact,
            )
            st.caption(f"Neighbor count comes from config.yaml: k = {base_cfg.retrieval.k}.")

        run = st.button("Run pipeline", type="primary", width="stretch")

    try:
        cfg = _run_config(
            base_cfg,
            live=live_execution,
            use_projected_contact=use_projected_contact,
            semantic=semantic_w,
            mass=mass_w,
            roughness=roughness_w,
            contact=contact_w,
            sigma_mass=sigma_mass,
            sigma_contact=sigma_contact,
        )
    except ValueError as error:
        with output:
            st.error(str(error))
        return

    if run:
        counterfactual = bool(
            uploaded is not None
            or not np.isclose(mass, sample.mass_g)
            or roughness != sample.roughness_class
            or not np.isclose(contact, sample.projected_contact_fraction)
        )
        training = (
            records
            if counterfactual
            else [record for record in records if record.object_id != object_id]
        )
        prepared_item = prepared.get(object_id)
        prepared_text = (
            prepared_item.descriptor.description if prepared_item is not None else None
        )
        semantic_description = (
            None if uploaded is not None else prepared_text or sample.semantic_description
        )
        with output, st.spinner("Running the shared pipeline..."):
            pipe = Pipeline(cfg, experiment).fit(training)
            detailed = pipe.predict_detailed(
                QueryInput(
                    object_id=f"custom_{object_id}" if counterfactual else object_id,
                    mass_g=mass,
                    roughness_class=roughness,
                    projected_contact_fraction=contact,
                    image_bgr=query_image,
                    image_path=sample.image_path if uploaded is None else "",
                    semantic_description=semantic_description,
                )
            )
            baseline = None
            if counterfactual:
                baseline_pipe = Pipeline(cfg, experiment).fit(
                    [record for record in records if record.object_id != object_id]
                )
                baseline = baseline_pipe.predict_detailed(
                    QueryInput(
                        object_id=object_id,
                        mass_g=sample.mass_g,
                        roughness_class=sample.roughness_class,
                        projected_contact_fraction=sample.projected_contact_fraction,
                        image_bgr=source_image,
                        image_path=sample.image_path,
                        semantic_description=prepared_text or sample.semantic_description,
                    )
                )
            run_path = save_pipeline_run(
                cfg,
                detailed=detailed,
                experiment=experiment,
                execution_mode=mode or "Offline",
                query={
                    "object_id": f"custom_{object_id}" if counterfactual else object_id,
                    "source_object_id": object_id,
                    "object_name": selected_name,
                    "mass_g": mass,
                    "roughness_class": roughness,
                    "projected_contact_fraction": contact,
                    "semantic_description": detailed.semantic_description,
                    "original_image_path": sample.image_path if uploaded is None else None,
                },
                truth=_truth_payload(truth),
                counterfactual=counterfactual,
                image_bgr=query_image,
                baseline=baseline,
            )
            st.session_state["single_result"] = (
                detailed,
                truth,
                counterfactual,
                baseline,
                cfg,
                experiment,
                run_path,
            )
            st.session_state["last_experiment"] = experiment

    with output:
        st.subheader("Pipeline output")
        if "single_result" not in st.session_state:
            st.info("Select a dataset object or upload an image, then run the pipeline.")
            _render_formula(cfg)
        else:
            (
                detailed,
                stored_truth,
                counterfactual,
                baseline,
                stored_cfg,
                stored_experiment,
                run_path,
            ) = st.session_state["single_result"]
            _render_prediction(
                detailed,
                stored_truth,
                counterfactual=counterfactual,
                baseline=baseline,
                cfg=stored_cfg,
                experiment=stored_experiment,
            )
            st.caption(f"Saved run: {run_path.name}")


def benchmark_view(base_cfg: Config) -> None:
    left, right = st.columns([0.28, 0.72], gap="large")
    with left:
        st.subheader("Benchmark run")
        experiment = st.selectbox(
            "Experiment",
            list(EXPERIMENT_IDS),
            index=2,
            format_func=_format_experiment,
            key="benchmark_experiment",
        )
        mode = st.segmented_control(
            "Execution", ["Offline", "Live Gemini"], default="Offline", key="benchmark_mode"
        )
        run = st.button("Run 129-object leave-one-out benchmark", type="primary", width="stretch")
        st.caption("Results are synthetic pipeline diagnostics, not real-world performance claims.")

    if run:
        cfg = base_cfg.model_copy(deep=True)
        cfg.models.dry_run = mode != "Live Gemini"
        progress_bar = right.progress(0.0)
        status = right.empty()

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total)
            status.caption(f"{done}/{total}: {name.replace('_', ' ')}")

        with right, st.spinner("Evaluating all objects with leave-one-out training..."):
            benchmark = run_benchmark(cfg, experiment, progress=progress)
            paths = save_benchmark(cfg, benchmark)
            st.session_state["benchmark_result"] = (benchmark, paths)
        progress_bar.empty()
        status.empty()

    with right:
        st.subheader("Leave-one-out results")
        if "benchmark_result" not in st.session_state:
            st.info("Run all 129 objects with each query excluded from its own training set.")
            return
        benchmark, paths = st.session_state["benchmark_result"]
        force = benchmark.metrics["force"]["overall"]
        selection = benchmark.metrics["selection"]
        recommendation = benchmark.metrics["model_recommendation"]
        metrics = st.columns(5)
        metrics[0].metric("Force MAE", f"{force.get('mae', float('nan')):.3f} N")
        metrics[1].metric("Force RMSE", f"{force.get('rmse', float('nan')):.3f} N")
        metrics[2].metric("Selection accuracy", f"{selection['accuracy']:.1%}")
        metrics[3].metric("Mean regret", f"{selection['mean_regret_n']:.3f} N")
        metrics[4].metric(
            "VLM recommendation accuracy",
            f"{recommendation['accuracy']:.1%}" if recommendation["n"] else "N/A",
        )
        result_frame = pd.DataFrame(benchmark.rows)
        plot_frame = pd.concat(
            [
                result_frame[["object_id", "true_gecko_force_n", "pred_gecko_force_n"]]
                .rename(columns={"true_gecko_force_n": "true", "pred_gecko_force_n": "pred"})
                .assign(gripper="gecko"),
                result_frame[["object_id", "true_silicone_force_n", "pred_silicone_force_n"]]
                .rename(columns={"true_silicone_force_n": "true", "pred_silicone_force_n": "pred"})
                .assign(gripper="silicone"),
            ],
            ignore_index=True,
        )
        st.scatter_chart(plot_frame, x="true", y="pred", color="gripper", size=70)
        display_columns = [
            "object_id", "mass_g", "roughness_class", "projected_contact_fraction",
            "true_gecko_force_n", "pred_gecko_force_n", "true_silicone_force_n",
            "pred_silicone_force_n", "true_favored", "predicted_gripper",
            "selection_correct", "regret_n",
            "model_recommended_gripper", "recommendation_agrees_with_selector",
        ]
        st.dataframe(result_frame[display_columns], hide_index=True, width="stretch")
        st.caption(f"Saved: {paths[0].name} and {paths[1].name}")


def _description_catalog(base_cfg: Config, rows: list) -> None:
    prepared = load_prepared_descriptors(base_cfg)
    st.info(
        "The source CSV does not contain curvature or physical contact area. "
        "Projected contact fraction is its only contact-geometry measurement. "
        "Every source column is shown on each object card below."
    )
    search = st.text_input("Search objects", placeholder="Material, object, condition...")
    page_size = st.segmented_control(
        "Objects per page", [8, 12, 24], default=12, key="catalog_page_size"
    )
    needle = search.strip().lower()
    filtered = []
    for row in rows:
        item = prepared.get(row.object_id)
        searchable = " ".join(
            (
                row.object_name,
                item.descriptor.description if item else "",
                item.descriptor.contact_material if item else "",
                item.descriptor.visible_surface_condition if item else "",
            )
        ).lower()
        if not needle or needle in searchable:
            filtered.append(row)

    size = int(page_size or 12)
    page_count = max(1, (len(filtered) + size - 1) // size)
    page = st.number_input(
        "Page", min_value=1, max_value=page_count, value=1, step=1, key="catalog_page"
    )
    start = (int(page) - 1) * size
    st.caption(
        f"Showing {start + 1 if filtered else 0}-{min(start + size, len(filtered))} "
        f"of {len(filtered)} objects. All 129 are available through search and paging."
    )

    for row in filtered[start : start + size]:
        item = prepared.get(row.object_id)
        image_path = base_cfg.root / f"data/expforce/images/{row.image_name}"
        with st.container(border=True):
            image_col, detail_col = st.columns([0.28, 0.72], gap="medium")
            with image_col:
                if image_path.exists():
                    image = _thumbnail(str(image_path), image_path.stat().st_mtime_ns)
                    if image is not None:
                        st.image(image, width="stretch")
                else:
                    st.warning("Image not downloaded")
            with detail_col:
                st.subheader(row.object_name)
                st.caption(f"Object ID: {row.object_id} | Source image: {row.image_name}")

                sensor_cols = st.columns(3)
                sensor_cols[0].metric("Mass", f"{row.mass_g:g} g")
                sensor_cols[1].metric("Roughness class", row.roughness_class)
                sensor_cols[2].metric(
                    "Projected contact fraction", f"{row.projected_contact_fraction:.3f}"
                )

                force_cols = st.columns(3)
                force_cols[0].metric(
                    "Gecko force",
                    f"{row.gecko_force_n:.2f} N" if row.gecko_force_n is not None else "None",
                )
                force_cols[0].caption(
                    f"Feasible: {'Yes' if row.gecko_feasible else 'No'}"
                )
                force_cols[1].metric(
                    "Silicone force",
                    (
                        f"{row.silicone_force_n:.2f} N"
                        if row.silicone_force_n is not None
                        else "None"
                    ),
                )
                force_cols[1].caption(
                    f"Feasible: {'Yes' if row.silicone_feasible else 'No'}"
                )
                force_cols[2].metric("Favored gripper", row.favored_gripper.title())

                if item is None:
                    st.warning("No descriptor checkpoint. Run live Data Preparation.")
                    continue
                source_label = item.descriptor_source.replace("_", " ").title()
                st.caption(
                    f"{source_label} | Embedding {item.embedding_status} | "
                    f"{item.embedding_model or 'not generated'}"
                )
                st.write(item.descriptor.description)
                st.markdown(
                    f"**Contact region:** {item.descriptor.contact_region}  \n"
                    f"**Contact material:** {item.descriptor.contact_material}  \n"
                    f"**Surface condition:** "
                    f"{item.descriptor.visible_surface_condition}  \n"
                    f"**Local geometry:** {item.descriptor.local_geometry}  \n"
                    f"**Uncertainty:** {item.descriptor.uncertainty}"
                )


def _pipeline_run_inspector(base_cfg: Config) -> None:
    runs = load_saved_runs(base_cfg)
    if not runs:
        st.info("No saved single runs yet. Run an object in Single Run first.")
        return

    labels = {
        (
            f"{run['created_at'][:19]} | {run['query'].get('object_name', run['query']['object_id'])} "
            f"| {run['experiment_display_name']} | {run['execution_mode']}"
        ): run
        for run in runs
    }
    selected = st.selectbox("Saved run", list(labels), key="saved_run_selector")
    run = labels[selected]
    query = run["query"]
    current_hash = source_sha256(base_cfg)
    if run.get("source_sha256") != current_hash:
        st.warning(
            "This run was produced from a different dataset version. Its saved truth is shown "
            "for provenance, but it is not rescored against the current data."
        )

    query_col, output_col = st.columns([0.34, 0.66], gap="large")
    with query_col:
        st.subheader("Exact query")
        image_rel = query.get("image_artifact_path") or query.get("original_image_path")
        image_path = base_cfg.root / image_rel if image_rel else None
        if image_path and image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.warning("Saved query image is unavailable.")
        st.caption(f"Image SHA-256: {query.get('image_sha256') or 'not recorded'}")
        st.metric("Mass", f"{query['mass_g']:.1f} g")
        sensor_cols = st.columns(2)
        sensor_cols[0].metric("Roughness", query["roughness_class"])
        sensor_cols[1].metric("Contact", f"{query['projected_contact_fraction']:.3f}")
        st.subheader("Run configuration")
        st.write(f"**Experiment:** {run['experiment_display_name']}")
        st.write(
            f"**Method/version:** {run.get('experiment_method', 'legacy')} / "
            f"{run.get('experiment_definition_version', 'legacy')}"
        )
        st.write(f"**Execution:** {run['execution_mode']}")
        st.write(f"**VLM:** {run['models']['vlm']}")
        st.write(f"**Text embedding:** {run['models']['embedding']}")
        st.write(f"**Protocol:** {run['evaluation_protocol'].replace('-', ' ')}")
        st.write(f"**Description:** {query.get('semantic_description', '')}")
        with st.expander("Retrieval parameters"):
            retrieval = run["retrieval_config"]
            saved_inputs = run.get("inputs", {})
            contact_enabled = saved_inputs.get(
                "use_projected_contact",
                retrieval.get("use_projected_contact", True),
            )
            st.write(f"Projected contact enabled: {contact_enabled}")
            st.write(f"Saved run top k: {retrieval['k']}")
            if retrieval["k"] != base_cfg.retrieval.k:
                st.info(
                    "This historical run used "
                    f"k={retrieval['k']}; the current config.yaml uses "
                    f"k={base_cfg.retrieval.k}. New runs use the current value."
                )
            st.write(f"Mass sigma: {retrieval['sigma_mass']}")
            st.write(f"Contact sigma: {retrieval['sigma_contact']}")
            st.json(retrieval["weights"], expanded=True)
        with st.expander("Experiment definition"):
            st.json(
                run.get("experiment_definition", run.get("experiment_toggles", {})),
                expanded=True,
            )
        truth = run.get("truth")
        if truth:
            st.subheader("Saved synthetic truth")
            st.write(
                f"Gecko: {truth.get('true_gecko_force_n')} N | "
                f"Silicone: {truth.get('true_silicone_force_n')} N | "
                f"Winner: {truth.get('true_selection', 'unknown').title()}"
            )
            if run.get("counterfactual"):
                st.caption("Context only: counterfactual runs are not scored against this truth.")

    with output_col:
        st.subheader("Pipeline output")
        cfg = base_cfg.model_copy(deep=True)
        cfg.models.dry_run = run["execution_mode"] != "Live Gemini"
        cfg.retrieval = type(cfg.retrieval).model_validate(run["retrieval_config"])
        cfg.inputs.use_projected_contact = run.get("inputs", {}).get(
            "use_projected_contact",
            run["retrieval_config"].get("use_projected_contact", True),
        )
        detailed = pipeline_result_from_dict(run["result"])
        baseline = (
            pipeline_result_from_dict(run["baseline"]) if run.get("baseline") else None
        )
        records = load_experience_pool(base_cfg)
        objects = group_by_object(records)
        source_id = query.get("source_object_id")
        truth_obj = objects.get(source_id)
        score_as_current = (
            truth_obj is not None
            and run.get("source_sha256") == current_hash
            and not run.get("counterfactual", False)
        )
        _render_prediction(
            detailed,
            truth_obj,
            counterfactual=not score_as_current,
            baseline=baseline,
            cfg=cfg,
            experiment=run["experiment_display_name"],
        )


def data_viewer(base_cfg: Config, rows: list) -> None:
    st.header("Data Viewer")
    view = st.segmented_control(
        "View",
        ["Description Catalog", "Pipeline Run Inspector"],
        default="Description Catalog",
        key="data_viewer_mode",
    )
    if view == "Pipeline Run Inspector":
        _pipeline_run_inspector(base_cfg)
    else:
        _description_catalog(base_cfg, rows)


def preparation_view(base_cfg: Config, summary: dict) -> None:
    st.subheader("What Data Preparation does")
    st.write(
        "This step validates all 129 synthetic objects and writes 258 gripper-specific "
        "experience rows. Live preparation downloads images, checkpoints one structured "
        "contact-region descriptor per object, and warms one text embedding per object. "
        "It does not run force prediction."
    )
    st.info(
        "Each Gemini descriptor is saved immediately. If quota interrupts preparation, rerun "
        "the same button to resume from the saved checkpoints and content-addressed cache."
    )
    stats = st.columns(4)
    stats[0].metric("Source objects", summary["objects"])
    stats[1].metric("Experience-pool objects", summary["objects"])
    stats[2].metric("Known-object training", f"{summary['objects'] - 1} per run")
    stats[3].metric("Experience rows", summary["experience_rows"])
    st.code(summary["source_sha256"], language=None)
    distributions = [
        {"dimension": "Roughness", "category": str(category), "count": count}
        for category, count in summary["roughness_counts"].items()
    ] + [
        {"dimension": "Favored gripper", "category": category, "count": count}
        for category, count in summary["favored_counts"].items()
    ]
    st.dataframe(pd.DataFrame(distributions), hide_index=True, width="stretch")

    left, right = st.columns(2)
    offline = left.button("Build records from checkpoints", width="stretch")
    live = right.button(
        "Download images + Gemini descriptors", type="primary", width="stretch"
    )
    if offline or live:
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total)
            status.caption(f"{done}/{total}: {name}")

        try:
            manifest = prepare_dataset(base_cfg, live=live, progress=progress)
        except Exception as error:  # noqa: BLE001 - surface preparation failures in the UI
            st.error(f"Preparation failed: {error}")
        else:
            st.session_state["preparation_manifest"] = manifest
            if manifest["missing_images"]:
                st.warning(f"Missing {len(manifest['missing_images'])} images.")
            elif live:
                st.success("All 129 descriptors, text embeddings, and 258 rows are prepared.")
            else:
                st.success("All 258 experience rows were rebuilt from available checkpoints.")
        progress_bar.empty()
        status.empty()

    manifest_path = base_cfg.root / PREPARATION_RELATIVE
    if manifest_path.exists():
        st.json(json.loads(manifest_path.read_text()), expanded=False)


def cache_view(base_cfg: Config) -> None:
    cache_dir = base_cfg.path("cache")
    files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
    result_dir = base_cfg.root / RESULTS_RELATIVE
    results = list(result_dir.glob("*")) if result_dir.exists() else []
    runs = load_saved_runs(base_cfg)
    stats = st.columns(4)
    stats[0].metric("Cached API responses", len(files))
    stats[1].metric("Cache size", f"{sum(path.stat().st_size for path in files) / 1024:.1f} KB")
    stats[2].metric("Saved benchmark files", len(results))
    stats[3].metric("Saved single runs", len(runs))
    st.code(str(cache_dir), language=None)
    if "single_result" in st.session_state:
        detailed = st.session_state["single_result"][0]
        st.subheader("Latest run telemetry")
        st.json(detailed.cache_stats or {"mode": "offline", "backend_attempts": 0})
    st.caption(
        "Cache keys include model, prompt, schema, image bytes, embedding dimension, and full prediction payload."
    )


def help_view(base_cfg: Config) -> None:
    st.header("How to use the lab")
    st.markdown(
        """
1. Open **Data Preparation** and build the derived records. Use the live option when you
   want image-derived contact descriptions and reference text embeddings.
2. Open **Single Run**, choose any of the 129 objects, or upload an image.
3. Set mass, roughness, and projected contact fraction. Leaving the original values scores
   with that object excluded from training; changing a value or image creates an unscored
   counterfactual that uses the full experience pool.
4. Choose an experiment and **Offline** or **Live Gemini**. Adjust retrieval weights and
   similarity constants when the selected experiment uses retrieval.
5. Click **Run pipeline**. Read the selected gripper and force first, then compare both
   predictions, evidence summaries, and the top-five matches.
6. Use **129-Object Benchmark** for leave-one-object-out evaluation. Use **Data Viewer**
   to inspect every image/description and saved run, and use
   **Cache Status** to confirm that repeated Gemini requests are being reused.
        """
    )

    st.header("Experiment definitions")
    uses = {
        "e1": "Image + joint VLM",
        "e2": "Image + sensors + joint VLM",
        "e4": "Sensors + paired retrieval + joint VLM",
        "e5": "Sensors + calibrated physics",
        "e6": "Sensors + calibrated physics + semantic residual",
    }
    experiments = pd.DataFrame(
        [
            {
                "Experiment": experiment_id.upper(),
                "Uses": uses[experiment_id],
                "Meaning": EXPERIMENT_CATALOG[experiment_id].summary,
            }
            for experiment_id in EXPERIMENT_IDS
        ]
    )
    st.dataframe(experiments, hide_index=True, width="stretch")
    st.caption(
        "Useful comparisons: E1 vs E2 isolates authoritative measurements; E2 vs E4 "
        "isolates paired experience retrieval; E5 vs E6 isolates the learned residual."
    )

    st.subheader("Prediction prompt routing")
    st.caption(
        "Only VLM force-prediction experiments have prompts. All text below is loaded "
        "directly from config.yaml."
    )
    with st.expander("Shared prediction system prompt"):
        st.code(base_cfg.prompts.prediction_system, language=None)
    for experiment in ("e1", "e2", "e4"):
        prompt_key = base_cfg.experiment(experiment).prompt
        assert prompt_key is not None
        with st.expander(f"{experiment.upper()} instruction — prompts.experiments.{prompt_key}"):
            st.code(base_cfg.prompts.experiments[prompt_key], language=None)
    st.info("E5 and E6 do not call a VLM for force prediction, so they have no prompt.")

    st.subheader("How E6 learns the physics residual")
    st.markdown(
        """
E6 calibrates the same physics model as E5 on each training fold, then learns the target
`measured force − physics force` separately for gecko and silicone. The default
gradient-boosted trees use log mass, roughness, projected contact fraction, physics
force, and PCA-reduced semantic embedding features. At inference the continuous output
is `physics force + predicted residual`, clamped only to the 0–8 N hardware range.
E6 retrieves no neighbors and makes no VLM force-prediction call.
        """
    )

    st.header("What E4 + Live Gemini means")
    st.markdown(
        """
For a known dataset object, the pipeline excludes that object and uses the other 128 objects.
For a custom query, it uses all 129. It reuses one cached text embedding per reference object for
both grippers, embeds the query description, and ranks references with the displayed hybrid
similarity. It retrieves five objects once, with both gecko and silicone outcomes attached to
each object. One Gemini request returns both force estimates and an explicit recommendation.
E4 does not construct
or send a physics estimate. Python selects the lower feasible
predicted force and explicitly reports a prediction tie only when the continuous estimates are equal.

**Live Gemini** means Gemini-backed descriptor, embedding, and VLM stages are allowed to make
network calls. A cached identical request makes no new call. It does not mean the labels are
real, and it does not automatically replace an offline-prepared reference corpus with visual
descriptions; that is the job of live Data Preparation.
        """
    )

    manifest_path = base_cfg.root / PREPARATION_RELATIVE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        mode = manifest.get("descriptor_mode", "unknown")
        missing = len(manifest.get("missing_images", []))
        descriptors_done = manifest.get("descriptors_completed", 0)
        embeddings_done = manifest.get("embeddings_completed", 0)
        if manifest.get("status") == "complete" and missing == 0:
            st.success(
                f"Current experience pool: {descriptors_done} descriptors and "
                f"{embeddings_done} warmed reference text embeddings."
            )
        else:
            st.warning(
                f"Preparation status: {manifest.get('status', mode)}; {descriptors_done} "
                f"descriptors, {embeddings_done} embeddings, and {missing} unavailable images."
            )
    else:
        st.warning("No preparation manifest exists yet. Run Data Preparation before evaluation.")

    st.header("Application sections")
    sections = pd.DataFrame(
        [
            {
                "Section": "Single Run",
                "Purpose": "Inspect one leave-one-out or custom query with full evidence.",
            },
            {
                "Section": "129-Object Benchmark",
                "Purpose": "Evaluate every object using leave-one-object-out training.",
            },
            {
                "Section": "Data Viewer",
                "Purpose": "Browse all images/descriptions and inspect exact saved pipeline runs.",
            },
            {
                "Section": "Data Preparation",
                "Purpose": "Fetch images, checkpoint descriptions, create rows, and warm text embeddings.",
            },
            {
                "Section": "Cache Status",
                "Purpose": "Inspect saved API responses and telemetry from the latest single run.",
            },
            {
                "Section": "Help & Experiments",
                "Purpose": "Understand controls, experiment ablations, and result interpretation.",
            },
        ]
    )
    st.dataframe(sections, hide_index=True, width="stretch")
    st.info(
        "All forces, winner labels, and reported accuracy are synthetic pipeline-validation "
        "signals. They can reveal software or modeling behavior, but cannot establish physical "
        "gripper performance."
    )


def contact_area_view(cfg: Config) -> None:
    st.subheader("Contact-Area Capture")
    st.caption(
        "Capture a real object from the camera (the Orbbec's RGB stream shows "
        "up as a normal USB webcam), extract its outline, and estimate the "
        "finger-contact fraction. Each capture is saved under "
        "`data/test_contact_area/<name>/` with the image named `<name>.png`. "
        "If the wrong feed appears, change the camera index below."
    )

    p = st.columns(3)
    px_per_mm = p[0].number_input(
        "px per mm (required)", min_value=0.0, value=8.0, step=0.1,
        help="From a known-width fiducial in the scene. Every mm-valued "
             "parameter is meaningless without a real scale.",
    )
    k_max = p[1].number_input("k_max (1/mm)", min_value=0.01, value=2.0, step=0.5)
    delta = p[2].number_input("delta (mm)", min_value=0.0, value=0.3, step=0.1)

    with st.expander("Advanced parameters"):
        a = st.columns(3)
        L = a[0].number_input("pad length L (mm)", min_value=0.5, value=4.0, step=0.5)
        w_pad = a[1].number_input("pad width (mm)", min_value=0.5, value=12.0, step=1.0)
        sweep_str = a[2].text_input("k_max sweep (logged)", value="1,2,4")
        use_finger = st.checkbox(
            "Finger drop-depth model", value=False,
            help="Constrain the pad's height band by finger geometry instead "
                 "of free placement. Needed for short objects (fruit) whose "
                 "pad cannot reach the equator.",
        )
        finger = None
        if use_finger:
            g = st.columns(4)
            finger = _build_finger_geometry(
                finger_length=g[0].number_input(
                    "finger length (mm)", min_value=1.0, value=100.0),
                pad_length=L,
                pad_start=g[1].number_input(
                    "pad start (mm)", min_value=0.0, value=0.0),
                tip_clearance=g[2].number_input(
                    "tip clearance (mm)", min_value=0.0, value=2.0),
                palm_standoff=g[3].number_input(
                    "palm standoff (mm)", min_value=0.0, value=5.0),
            )

    st.divider()
    name_in = st.text_input("Object name", placeholder="e.g. water_bottle")
    controls = st.columns([1, 1, 1, 1])
    cam_index = int(controls[0].number_input(
        "camera index", min_value=0, value=0, step=1,
        help="Which USB video device to use. The Orbbec RGB feed is often 0 "
             "or 1; change it if the wrong camera appears."))
    live = controls[1].toggle("Live preview", value=False)
    capture = controls[2].button("Capture & Analyze", type="primary")
    overwrite = controls[3].checkbox("Overwrite if name exists", value=False)

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
        run_dir = TEST_CONTACT_ROOT / name
        if run_dir.exists() and not overwrite:
            st.error(f"`{run_dir.name}` already exists. Enable 'Overwrite' or "
                     "choose another name.")
            st.stop()

        try:
            frame = _read_camera_frame(cam_index)
        except Exception as exc:
            st.error("Could not read a frame from the camera. Check the USB "
                     "connection, that no other process holds it, and that the "
                     "camera index is correct.")
            st.exception(exc)
            st.stop()

        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"{name}.png"
        cv2.imwrite(str(image_path), frame)

        from pipeline_core import ContactParams, analyze_image

        params = ContactParams(
            px_per_mm=px_per_mm, object_type=CONTACT_OBJECT_TYPE,
            closing_axis=CONTACT_CLOSING_AXIS, k_max=k_max, delta=delta,
            L=L, w_pad=w_pad,
            sweep_k=tuple(float(v) for v in sweep_str.split(",") if v.strip()),
            finger=finger,
        )
        try:
            with st.spinner("Extracting outline and computing contact area..."):
                est, summary, paths = analyze_image(
                    image_path, run_dir, name, params,
                    session=_rembg_session(),
                    index_csv=TEST_CONTACT_ROOT / "index.csv",
                )
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

        if res["summary"].get("finger") and not res["feasible"]:
            st.error("Grasp INFEASIBLE: the pad's height band sits above the "
                     "object top. Reported contact is zero.")

        m = st.columns(4)
        m[0].metric("Mean contact fraction", f"{r['mean_fraction']:.3f}")
        m[1].metric("Total area", f"{r['total_area_mm2']:.1f} mm2")
        m[2].metric(
            "Contact L / R",
            f"{r['left']['contact_mm']:.1f} / {r['right']['contact_mm']:.1f} mm")
        m[3].metric("Antipodal grasp", "yes" if r["antipodal_grasp"] else "no")

        cols = st.columns(2)
        cols[0].image(res["paths"]["contact_fig"],
                      caption="Contact model (numbers at top)",
                      use_container_width=True)
        cols[1].image(res["paths"]["spline_overlay"],
                      caption="Fitted outline over the capture",
                      use_container_width=True)

        st.caption("k_max sweep (mean fraction): "
                   f"{res['summary']['k_max_sweep_mean_fraction']}")
        with st.expander("summary.json"):
            st.json(res["summary"])
        st.success(f"Saved to `{res['run_dir']}`")


def _build_finger_geometry(**kwargs):
    """Lazy import keeps rembg/contact_model off the app-startup path."""
    from contact_area import FingerGeometry

    return FingerGeometry(**kwargs)


def main() -> None:
    base_cfg = load_config().model_copy(deep=True)
    rows, summary = _load_static()
    st.title("Force Pipeline Lab")
    st.markdown(
        '<div class="synthetic-note"><b>Synthetic pipeline validation.</b> '
        'These results test software behavior and model integration, not physical gripper performance.</div>',
        unsafe_allow_html=True,
    )
    (
        single_tab, benchmark_tab, viewer_tab, contact_tab, preparation_tab,
        cache_tab, help_tab,
    ) = st.tabs(
        [
            "Single Run",
            "129-Object Benchmark",
            "Data Viewer",
            "Contact Area",
            "Data Preparation",
            "Cache Status",
            "Help & Experiments",
        ]
    )
    with single_tab:
        single_run_view(base_cfg, rows)
    with benchmark_tab:
        benchmark_view(base_cfg)
    with viewer_tab:
        data_viewer(base_cfg, rows)
    with contact_tab:
        contact_area_view(base_cfg)
    with preparation_tab:
        preparation_view(base_cfg, summary)
    with cache_tab:
        cache_view(base_cfg)
    with help_tab:
        help_view(base_cfg)


if __name__ == "__main__":
    main()
