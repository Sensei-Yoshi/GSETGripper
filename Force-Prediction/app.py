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
    get_or_create_split,
    load_image,
    load_rows,
    load_validation_records,
    prepare_dataset,
    run_benchmark,
    save_benchmark,
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


@st.cache_data(show_spinner=False)
def _load_static() -> tuple[list, dict, dict]:
    cfg = load_config().model_copy(deep=True)
    rows = load_rows(cfg)
    split = get_or_create_split(cfg)
    return rows, split, validation_summary(cfg, rows)


def _run_config(
    base: Config,
    *,
    live: bool,
    semantic: float,
    mass: float,
    roughness: float,
    contact: float,
    sigma_mass: float,
    sigma_contact: float,
) -> Config:
    cfg = base.model_copy(deep=True)
    cfg.models.dry_run = not live
    cfg.retrieval.k = 7
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


def _rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr is not None else None


def _truth_for_display(obj) -> dict:  # noqa: ANN001
    optimal, _ = obj.optimal_grippers()
    return {
        "true_gecko_force_n": obj.gecko.min_force_n if obj.gecko else None,
        "true_silicone_force_n": obj.silicone.min_force_n if obj.silicone else None,
        "true_selection": "tie" if len(optimal) > 1 else next(iter(optimal)).value,
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
) -> None:
    result = detailed.selection
    metric_cols = st.columns(3)
    metric_cols[0].metric("Selected gripper", result.desired_gripper.title())
    metric_cols[1].metric(
        "Selected force",
        f"{result.predicted_normal_force_n:.2f} N" if result.predicted_normal_force_n else "None",
    )
    metric_cols[2].metric("Experiment", st.session_state.get("last_experiment", "E5").upper())

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
            st.metric("Selected-force change from baseline", f"{delta:+.2f} N")
    else:
        truth_values = _truth_for_display(truth)
        predicted_correct = (
            result.desired_gripper in {g.value for g in truth.optimal_grippers()[0]}
        )
        st.markdown(
            f'<p class="status-ok">Held-out selection: {truth_values["true_selection"]}; '
            f'prediction {"correct" if predicted_correct else "incorrect"}.</p>',
            unsafe_allow_html=True,
        )

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
                f"{pred.predicted_normal_force_n:.2f} N",
                delta=(
                    f"{pred.predicted_normal_force_n - truth_force:+.2f} N error"
                    if truth_force is not None
                    else None
                ),
                delta_color="inverse",
            )
            physics = detailed.physics_estimates.get(gripper)
            st.caption(
                f"Physics prior: {physics.get('min_force_n')} N"
                if physics
                else "Physics prior: not used"
            )
            st.write(pred.reasoning_trace or "No reasoning trace for this experiment.")

    st.subheader("Top seven reference matches")
    if any(detailed.retrieved.values()):
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


def single_run_view(base_cfg: Config, rows: list, split: dict) -> None:
    records = load_validation_records(base_cfg)
    objects = group_by_object(records)
    row_by_id = {row.object_id: row for row in rows}
    test_ids = split["test_object_ids"]
    names = {row_by_id[object_id].object_name: object_id for object_id in test_ids}

    controls, output = st.columns([0.38, 0.62], gap="large")
    with controls:
        st.subheader("Query")
        selected_name = st.selectbox("Held-out object", sorted(names))
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

        with st.expander("Retrieval tuning", expanded=True):
            semantic_w = st.slider("Semantic weight", 0.0, 1.0, 0.40, 0.05)
            mass_w = st.slider("Mass weight", 0.0, 1.0, 0.25, 0.05)
            roughness_w = st.slider("Roughness weight", 0.0, 1.0, 0.20, 0.05)
            contact_w = st.slider("Contact weight", 0.0, 1.0, 0.15, 0.05)
            sigma_mass = st.slider("Mass sigma", 0.1, 3.0, 0.7, 0.1)
            sigma_contact = st.slider("Contact sigma", 0.05, 1.0, 0.25, 0.05)

        run = st.button("Run pipeline", type="primary", width="stretch")

    try:
        cfg = _run_config(
            base_cfg,
            live=mode == "Live Gemini",
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
        reference_ids = set(split["reference_object_ids"])
        training = [record for record in records if record.object_id in reference_ids]
        counterfactual = bool(
            uploaded is not None
            or not np.isclose(mass, sample.mass_g)
            or roughness != sample.roughness_class
            or not np.isclose(contact, sample.projected_contact_fraction)
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
                    semantic_description=None if uploaded is not None else sample.semantic_description,
                )
            )
            baseline = None
            if counterfactual:
                baseline = pipe.predict_detailed(
                    QueryInput(
                        object_id=object_id,
                        mass_g=sample.mass_g,
                        roughness_class=sample.roughness_class,
                        projected_contact_fraction=sample.projected_contact_fraction,
                        image_bgr=source_image,
                        image_path=sample.image_path,
                        semantic_description=sample.semantic_description,
                    )
                )
            st.session_state["single_result"] = (detailed, truth, counterfactual, baseline, cfg)
            st.session_state["last_experiment"] = experiment

    with output:
        st.subheader("Pipeline output")
        if "single_result" not in st.session_state:
            st.info("Select a held-out object or upload an image, then run the pipeline.")
            _render_formula(cfg)
        else:
            detailed, stored_truth, counterfactual, baseline, stored_cfg = st.session_state[
                "single_result"
            ]
            _render_prediction(
                detailed,
                stored_truth,
                counterfactual=counterfactual,
                baseline=baseline,
                cfg=stored_cfg,
            )


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
        run = st.button("Run 29-object benchmark", type="primary", width="stretch")
        st.caption("Results are synthetic pipeline diagnostics, not real-world performance claims.")

    if run:
        cfg = base_cfg.model_copy(deep=True)
        cfg.models.dry_run = mode != "Live Gemini"
        cfg.retrieval.k = 7
        progress_bar = right.progress(0.0)
        status = right.empty()

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total)
            status.caption(f"{done}/{total}: {name.replace('_', ' ')}")

        with right, st.spinner("Evaluating held-out objects..."):
            benchmark = run_benchmark(cfg, experiment, progress=progress)
            paths = save_benchmark(cfg, benchmark)
            st.session_state["benchmark_result"] = (benchmark, paths)
        progress_bar.empty()
        status.empty()

    with right:
        st.subheader("Held-out results")
        if "benchmark_result" not in st.session_state:
            st.info("Run the fixed 29-object holdout to populate metrics and plots.")
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


def preparation_view(base_cfg: Config, summary: dict, split: dict) -> None:
    stats = st.columns(4)
    stats[0].metric("Source objects", summary["objects"])
    stats[1].metric("Reference objects", len(split["reference_object_ids"]))
    stats[2].metric("Held-out objects", len(split["test_object_ids"]))
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
    offline = left.button("Prepare derived data offline", width="stretch")
    live = right.button("Download images and prepare live descriptors", type="primary", width="stretch")
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
            else:
                st.success("All 129 objects and 258 experience rows are prepared.")
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
    stats = st.columns(3)
    stats[0].metric("Cached API responses", len(files))
    stats[1].metric("Cache size", f"{sum(path.stat().st_size for path in files) / 1024:.1f} KB")
    stats[2].metric("Saved benchmark files", len(results))
    st.code(str(cache_dir), language=None)
    if "single_result" in st.session_state:
        detailed = st.session_state["single_result"][0]
        st.subheader("Latest run telemetry")
        st.json(detailed.cache_stats or {"mode": "offline", "backend_attempts": 0})
    st.caption(
        "Cache keys include model, prompt, schema, image bytes, embedding dimension, and full prediction payload."
    )


def main() -> None:
    base_cfg = load_config().model_copy(deep=True)
    rows, split, summary = _load_static()
    st.title("Force Pipeline Lab")
    st.markdown(
        '<div class="synthetic-note"><b>Synthetic pipeline validation.</b> '
        'These results test software behavior and model integration, not physical gripper performance.</div>',
        unsafe_allow_html=True,
    )
    single_tab, benchmark_tab, preparation_tab, cache_tab = st.tabs(
        ["Single Run", "29-Object Benchmark", "Data Preparation", "Cache Status"]
    )
    with single_tab:
        single_run_view(base_cfg, rows, split)
    with benchmark_tab:
        benchmark_view(base_cfg)
    with preparation_tab:
        preparation_view(base_cfg, summary, split)
    with cache_tab:
        cache_view(base_cfg)


if __name__ == "__main__":
    main()
