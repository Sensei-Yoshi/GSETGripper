from __future__ import annotations

from force_prediction.config import load_config
from force_prediction.contracts import Gripper, Query, group_by_object
from force_prediction.hardware import fabricate_records
from force_prediction.pipeline import Pipeline, query_input_from_object
from force_prediction.retrieval import ExperienceIndex, build_embedding_text


def test_paired_delta_matches_truth():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 20)
    objects = group_by_object(records)
    index = ExperienceIndex(cfg).fit(records)
    q = Query(object_id="probe", image_path="", mass_g=300, roughness_class=2,
              projected_contact_fraction=0.8, semantic_description="x")
    qv = index.provider.embed(build_embedding_text("x", 300, 2, 0.8, cfg))
    for r in index.retrieve(q, qv, Gripper.GECKO):
        truth_other = objects[r.record.object_id].other_gripper_force(Gripper.GECKO)
        assert r.other_gripper_min_force_n == truth_other


def test_e5_pipeline_runs_end_to_end():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [r for r in records if r.object_id != held]
    test = [r for r in records if r.object_id == held]
    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train)
    result = pipe.predict(query_input_from_object(test, cfg))
    assert result.desired_gripper in ("gecko", "silicone", "none")
    assert set(result.candidate_predictions) == {"gecko", "silicone"}


def test_detailed_pipeline_preserves_selection_and_exposes_trace():
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    cfg.retrieval.k = 7
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [r for r in records if r.object_id != held]
    test = [r for r in records if r.object_id == held]
    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train)
    query = query_input_from_object(test, cfg)

    detailed = pipe.predict_detailed(query)
    ordinary = pipe.predict(query)

    assert detailed.selection == ordinary
    assert set(detailed.retrieved) == {"gecko", "silicone"}
    assert all(len(items) == 7 for items in detailed.retrieved.values())
    assert set(detailed.physics_estimates) == {"gecko", "silicone"}
