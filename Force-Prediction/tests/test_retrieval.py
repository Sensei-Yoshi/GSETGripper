from __future__ import annotations

from force_prediction.config import load_config
from force_prediction.contracts import Gripper, Query
from force_prediction.hardware import fabricate_records
from force_prediction.retrieval import (
    ExperienceIndex,
    build_embedding_text,
    normalized_weights,
    s_contact,
    s_mass,
    s_roughness,
)

CFG = load_config()


def test_similarity_bounds():
    assert s_mass(100, 100, CFG.retrieval.sigma_mass) == 1.0
    assert 0 <= s_mass(100, 900, CFG.retrieval.sigma_mass) <= 1
    assert s_contact(0.5, 0.5, CFG.retrieval.sigma_contact) == 1.0
    assert s_roughness(2, 2, CFG) == 1.0
    assert s_roughness(1, 5, CFG) == 0.0
    assert s_roughness(1, 2, CFG) == 0.75


def test_embedding_text_is_semantic_only():
    first = build_embedding_text("smooth glass cup", 100, 1, 0.9, CFG)
    second = build_embedding_text("smooth glass cup", 900, 5, 0.2, CFG)
    assert first == second == "smooth glass cup"


def test_weights_normalize_and_breakdown_sums_to_score():
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    weights = normalized_weights(cfg)
    assert sum(weights.values()) == 1.0
    records = fabricate_records(cfg, 10)
    index = ExperienceIndex(cfg).fit(records)
    probe = records[0]
    q = Query(object_id="probe", image_path="", mass_g=probe.mass_g,
              roughness_class=probe.roughness_class,
              projected_contact_fraction=probe.projected_contact_fraction,
              semantic_description=probe.semantic_description)
    result = index.retrieve(q, index.embed_query(q), Gripper.GECKO)[0]
    assert result.similarity is not None
    contributions = (
        result.similarity.semantic_contribution
        + result.similarity.mass_contribution
        + result.similarity.roughness_contribution
        + result.similarity.contact_contribution
    )
    assert abs(result.score - contributions) < 1e-9
    assert result.rank == 1


def test_branch_filter_and_topk_and_paired():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 30)
    index = ExperienceIndex(cfg).fit(records)
    probe = records[0]
    q = Query(object_id="probe", image_path="", mass_g=probe.mass_g,
              roughness_class=probe.roughness_class,
              projected_contact_fraction=probe.projected_contact_fraction,
              semantic_description=probe.semantic_description)
    qv = index.provider.embed(build_embedding_text(
        q.semantic_description, q.mass_g, q.roughness_class, q.projected_contact_fraction, cfg))

    out = index.retrieve(q, qv, Gripper.GECKO, exclude_object_id="probe")
    assert len(out) == cfg.retrieval.k
    assert all(r.record.gripper is Gripper.GECKO for r in out)          # branch filter
    assert all(r.record.object_id != "probe" for r in out)             # exclusion
    assert out == sorted(out, key=lambda r: r.score, reverse=True)      # sorted
    # paired-row delta available for objects tested on both grippers
    assert any(r.other_gripper_min_force_n is not None for r in out)


def test_payload_paired_toggle():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 10)
    index = ExperienceIndex(cfg).fit(records)
    q = Query(object_id="probe", image_path="", mass_g=200, roughness_class=2,
              projected_contact_fraction=0.7, semantic_description="x")
    qv = index.provider.embed("x")
    r = index.retrieve(q, qv, Gripper.SILICONE)[0]
    assert "other_gripper_min_force_n" in r.to_payload(include_paired=True)
    assert "other_gripper_min_force_n" not in r.to_payload(include_paired=False)


def test_object_retrieval_ranks_once_and_carries_both_gripper_labels():
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    cfg.retrieval.k = 5
    records = fabricate_records(cfg, 20)
    index = ExperienceIndex(cfg).fit(records)
    probe = records[0]
    query = Query(
        object_id=probe.object_id,
        image_path="",
        mass_g=probe.mass_g,
        roughness_class=probe.roughness_class,
        projected_contact_fraction=probe.projected_contact_fraction,
        semantic_description=probe.semantic_description,
    )

    results = index.retrieve_objects(
        query,
        index.embed_query(query),
        exclude_object_id=probe.object_id,
    )

    assert len(results) == 5
    assert len({item.object_id for item in results}) == 5
    assert all(item.object_id != probe.object_id for item in results)
    assert all(item.gecko_min_force_n is not None for item in results)
    assert all(item.silicone_min_force_n is not None for item in results)
    assert results == sorted(results, key=lambda item: (-item.score, item.object_id))
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]
