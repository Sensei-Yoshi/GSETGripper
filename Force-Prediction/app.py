"""Streamlit research viewer for the synthetic two-gripper Exp-Force pipeline."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from force_prediction.config import Config, load_config
from force_prediction.contracts import Gripper, group_by_object
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
    cfg.retrieval.use_projected_contact = use_projected_contact
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


def _retrieval_table(result: PipelineRunResult, gripper: str) -> pd.DataFrame:
    rows = []
    for item in result.retrieved.get(gripper, []):
        sim = item.similarity
        rows.append(
            {
                "rank": item.rank,
                "object": item.record.object_id.replace("_", " "),
                "score": item.score,
                "semantic": sim.semantic if sim else None,
                "mass": sim.mass if sim else None,
                "roughness": sim.roughness if sim else None,
                "contact": sim.contact if sim else None,
                "mass_g": item.record.mass_g,
                "roughness_class": item.record.roughness_class,
                "contact_fraction": item.record.projected_contact_fraction,
                f"{gripper}_force_n": item.record.min_force_n,
                "paired_force_n": item.other_gripper_min_force_n,
            }
        )
    return pd.DataFrame(rows)


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
    metric_cols = st.columns(3)
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
        "Experiment", (experiment or st.session_state.get("last_experiment", "E5")).upper()
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
                "E5 retrieves each object once with both gripper outcomes. Offline mode "
                "uses the deterministic paired-neighbor stand-in and makes no VLM request."
            )
        else:
            st.caption(
                "E5 retrieves each object once. Both gripper outcomes come from the same "
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
    elif any(detailed.retrieved.values()):
        gecko_tab, silicone_tab = st.tabs(["Gecko branch", "Silicone branch"])
        with gecko_tab:
            st.dataframe(
                _retrieval_table(detailed, "gecko"),
                hide_index=True,
                width="stretch",
                column_config={key: st.column_config.NumberColumn(format="%.3f") for key in (
                    "score", "semantic", "mass", "roughness", "contact"
                )},
            )
        with silicone_tab:
            st.dataframe(
                _retrieval_table(detailed, "silicone"),
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
            ["e5", "e3", "e3b", "e4", "e6", "e2", "e1"],
            format_func=lambda value: value.upper(),
        )
        mode = st.segmented_control("Execution", ["Offline", "Live Gemini"], default="Offline")
        live_execution = mode == "Live Gemini"

        with st.expander("Retrieval tuning", expanded=True):
            use_projected_contact = st.checkbox(
                "Use projected contact fraction",
                value=base_cfg.retrieval.use_projected_contact,
                help=(
                    "When disabled, contact is omitted from VLM inputs and its retrieval "
                    "weight is set to zero; the remaining weights are renormalized. "
                    "Physics-based E4 and E6 still require the measured contact fraction."
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
            pipe = Pipeline(cfg, cfg.experiment(experiment)).fit(training)
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
                baseline_pipe = Pipeline(cfg, cfg.experiment(experiment)).fit(
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
            ["e5", "e3", "e3b", "e4", "e6", "e2", "e1"],
            format_func=str.upper,
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
        metrics = st.columns(4)
        metrics[0].metric("Force MAE", f"{force.get('mae', float('nan')):.3f} N")
        metrics[1].metric("Force RMSE", f"{force.get('rmse', float('nan')):.3f} N")
        metrics[2].metric("Selection accuracy", f"{selection['accuracy']:.1%}")
        metrics[3].metric("Mean regret", f"{selection['mean_regret_n']:.3f} N")
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
            f"| {run['experiment'].upper()} | {run['execution_mode']}"
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
        st.write(f"**Experiment:** {run['experiment'].upper()}")
        st.write(f"**Execution:** {run['execution_mode']}")
        st.write(f"**VLM:** {run['models']['vlm']}")
        st.write(f"**Text embedding:** {run['models']['embedding']}")
        st.write(f"**Protocol:** {run['evaluation_protocol'].replace('-', ' ')}")
        st.write(f"**Description:** {query.get('semantic_description', '')}")
        with st.expander("Retrieval parameters"):
            retrieval = run["retrieval_config"]
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
        with st.expander("Experiment toggles"):
            st.json(run["experiment_toggles"], expanded=True)
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
            experiment=run["experiment"],
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
    experiments = pd.DataFrame(
        [
            {
                "Experiment": "E1",
                "Uses": "Image + VLM",
                "Meaning": "Vision-only zero-shot baseline; measured sensors, retrieval, and physics are hidden.",
            },
            {
                "Experiment": "E2",
                "Uses": "Sensors + VLM",
                "Meaning": "Tests whether authoritative mass, roughness, and contact improve the VLM.",
            },
            {
                "Experiment": "E3",
                "Uses": "Sensors + retrieval + VLM",
                "Meaning": "Adds top-k same-gripper experiences, without paired rows or physics.",
            },
            {
                "Experiment": "E3b",
                "Uses": "Sensors + retrieval",
                "Meaning": "Pure similarity-weighted retrieval baseline; no VLM and no physics.",
            },
            {
                "Experiment": "E4",
                "Uses": "Sensors + physics",
                "Meaning": "Calibrated physics-only baseline; no retrieval and no VLM.",
            },
            {
                "Experiment": "E5",
                "Uses": "Sensors + paired-object retrieval + one VLM call",
                "Meaning": "Retrieves objects once with both force labels. Python selects the lower feasible prediction.",
            },
            {
                "Experiment": "E6",
                "Uses": "Sensors + physics + learned residual",
                "Meaning": "Classical learned correction to physics; no VLM decision or retrieval list.",
            },
        ]
    )
    st.dataframe(experiments, hide_index=True, width="stretch")
    st.caption(
        "Useful comparisons: E5 vs E3 tests the complete paired-object method against "
        "same-gripper experiential retrieval; E3 vs E3b tests the VLM contribution; "
        "E4 vs E6 tests the learned residual over physics."
    )

    st.subheader("Prediction prompt routing")
    st.caption(
        "Only VLM force-prediction experiments have prompts. All text below is loaded "
        "directly from config.yaml."
    )
    with st.expander("Shared prediction system prompt"):
        st.code(base_cfg.prompts.prediction_system, language=None)
    for experiment in ("e1", "e2", "e3", "e5"):
        prompt_key = base_cfg.experiment(experiment).prompt
        assert prompt_key is not None
        with st.expander(f"{experiment.upper()} instruction — prompts.experiments.{prompt_key}"):
            st.code(base_cfg.prompts.experiments[prompt_key], language=None)
    st.info("E3b, E4, and E6 do not call a VLM for force prediction, so they have no prompt.")

    st.subheader("How E6 learns the physics residual")
    st.markdown(
        """
E6 calibrates the E4 physics model on each training fold, then learns the target
`measured force − physics force` separately for gecko and silicone. The default
gradient-boosted trees use log mass, roughness, projected contact fraction, physics
force, and PCA-reduced semantic embedding features. At inference the continuous output
is `physics force + predicted residual`, clamped only to the 0–8 N hardware range.
E6 retrieves no neighbors and makes no VLM force-prediction call.
        """
    )

    st.header("What E5 + Live Gemini means")
    st.markdown(
        """
For a known dataset object, the pipeline excludes that object and uses the other 128 objects.
For a custom query, it uses all 129. It reuses one cached text embedding per reference object for
both grippers, embeds the query description, and ranks references with the displayed hybrid
similarity. It retrieves five objects once, with both gecko and silicone outcomes attached to
each object. One Gemini request returns the two structured force estimates. E5 does not construct
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


def main() -> None:
    base_cfg = load_config().model_copy(deep=True)
    rows, summary = _load_static()
    st.title("Force Pipeline Lab")
    st.markdown(
        '<div class="synthetic-note"><b>Synthetic pipeline validation.</b> '
        'These results test software behavior and model integration, not physical gripper performance.</div>',
        unsafe_allow_html=True,
    )
    single_tab, benchmark_tab, viewer_tab, preparation_tab, cache_tab, help_tab = st.tabs(
        [
            "Single Run",
            "129-Object Benchmark",
            "Data Viewer",
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
    with preparation_tab:
        preparation_view(base_cfg, summary)
    with cache_tab:
        cache_view(base_cfg)
    with help_tab:
        help_view(base_cfg)


if __name__ == "__main__":
    main()
