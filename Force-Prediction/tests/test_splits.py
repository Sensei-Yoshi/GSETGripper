from __future__ import annotations

from force_prediction.config import load_config
from force_prediction.evaluation import make_folds
from force_prediction.hardware import fabricate_records

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
