"""Run one (or all) experiments under frozen GroupKFold splits and print metrics.

    python scripts/run_experiment.py --exp e6
    python scripts/run_experiment.py --all

Leak-safe by construction: the pipeline is re-fit on each fold's training objects
(the retrieval index is fold-local), and both
gripper rows of an object always share a fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import EXPERIMENT_IDS, load_config  # noqa: E402
from modules.contracts import group_by_object, load_experiences  # noqa: E402
from modules.evaluation import (  # noqa: E402
    EvalRow,
    compute_metrics,
    freeze_splits,
    load_splits,
    make_folds,
)
from modules.pipeline import Pipeline, query_input_from_object  # noqa: E402


def get_or_make_splits(records, cfg, refresh: bool):
    path = cfg.path("splits")
    ids = {r.object_id for r in records}
    if path.exists() and not refresh:
        folds = load_splits(path)
        covered = {oid for f in folds for oid in f["train"] + f["test"]}
        if ids <= covered:
            return folds
    folds = make_folds(records, cfg)
    freeze_splits(folds, path)
    return folds


def run_experiment(name: str, records, cfg, folds) -> dict:
    by_object = group_by_object(records)
    rows: list[EvalRow] = []
    for fold in folds:
        train = [r for r in records if r.object_id in set(fold["train"])]
        pipe = Pipeline(cfg, name).fit(train)
        for oid in fold["test"]:
            obj_records = [r for r in records if r.object_id == oid]
            if not obj_records:
                continue
            result = pipe.predict(query_input_from_object(obj_records, cfg))
            rows.append(EvalRow(object_id=oid, truth=by_object[oid], result=result))
    return compute_metrics(rows, cfg).to_dict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", choices=EXPERIMENT_IDS, default="e6", help="Experiment ID.")
    p.add_argument("--all", action="store_true", help="Run every experiment.")
    p.add_argument(
        "--confirm-gemini-cost",
        action="store_true",
        help="Required for experiments that make Gemini generation or embedding calls.",
    )
    p.add_argument("--refresh-splits", action="store_true", help="Regenerate splits.json.")
    p.add_argument("--out", default=None, help="Write results JSON here.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    names = EXPERIMENT_IDS if args.all else [args.exp]
    if not args.confirm_gemini_cost:
        raise SystemExit(
            "active experiment evaluation requires --confirm-gemini-cost"
        )
    records = load_experiences(cfg.path("experiences"))
    if not records:
        print("No prepared experience records found for the configured dataset.")
        return 1
    folds = get_or_make_splits(records, cfg, args.refresh_splits)

    results = {}
    for name in names:
        metrics = run_experiment(name, records, cfg, folds)
        results[name] = metrics
        f = metrics["force"]["overall"]
        s = metrics["selection"]
        print(
            f"[{name:4}] force MAE={f.get('mae', float('nan')):.3f}N "
            f"(n={f.get('n', 0)})  sel acc={s.get('accuracy', float('nan')):.2f} "
            f"mean_regret={s.get('mean_regret_n', float('nan')):.3f}N"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
