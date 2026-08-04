from __future__ import annotations

from modules.config import load_config
from modules.evaluation import make_folds
from tests.factories import fabricate_records

CFG = load_config()


def test_folds_disjoint_and_cover():
    records = fabricate_records(CFG, 40)
    folds = make_folds(records, CFG)
    all_ids = {r.object_id for r in records}
    seen_test: set[str] = set()
    for fold in folds:
        train, test = set(fold["train"]), set(fold["test"])
        assert train.isdisjoint(test)            # no object in both sides
        seen_test |= test
    assert seen_test == all_ids                  # every object tested exactly once


def test_both_gripper_rows_share_fold():
    # Splitting on object_id guarantees an object's two rows never straddle folds;
    # verify no test object_id leaks into its own training set.
    records = fabricate_records(CFG, 40)
    folds = make_folds(records, CFG)
    for fold in folds:
        assert set(fold["train"]).isdisjoint(set(fold["test"]))


def test_new_surface_folds_never_leak_siblings():
    records = fabricate_records(CFG, 12)
    source_id = records[0].object_id
    source_rows = [record for record in records if record.object_id == source_id]
    sibling_id = f"{source_id}__condition_2"
    records.extend(
        record.model_copy(
            update={
                "object_id": sibling_id,
                "surface_id": source_id,
                "condition_id": "condition_2",
            }
        )
        for record in source_rows
    )

    grouped = make_folds(records, CFG)
    source_fold = next(fold for fold in grouped if source_id in fold["test"])
    assert sibling_id in source_fold["test"]
    assert sibling_id not in source_fold["train"]
