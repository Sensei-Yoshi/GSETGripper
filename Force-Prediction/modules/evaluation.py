"""Leak-safe splitting + force / selection / regret metrics.

Both gripper rows of an object always share a fold (GroupKFold on object_id, with
optional coarser variant grouping). Splits are frozen to splits.json so every
experiment is scored on identical folds.

Metrics
  force      : MAE, RMSE, medAE, %within thresholds (per gripper + overall)
  feasibility: accuracy / precision / recall of the feasible flag (per gripper)
  selection  : gripper-choice accuracy, infeasible-pick rate, and REGRET
               R(o) = F*(o, chosen) - F*(o, oracle)  (mean / median / worst)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .config import Config
from .contracts import ExperienceRecord, Gripper, ObjectRecord, SelectionResult, group_by_object


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_folds(
    records: list[ExperienceRecord],
    cfg: Config,
    variant_groups: dict[str, str] | None = None,
) -> list[dict[str, list[str]]]:
    """GroupKFold over object_id (or a coarser variant group). Returns per-fold
    {"train": [object_id...], "test": [object_id...]}."""
    object_ids = sorted({r.object_id for r in records})
    record_by_id = {record.object_id: record for record in records}
    groups = [
        (variant_groups or {}).get(
            oid,
            (
                record_by_id[oid].surface_id or oid
            ),
        )
        for oid in object_ids
    ]
    n_splits = min(cfg.evaluation.n_folds, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    x = np.zeros((len(object_ids), 1))
    folds = []
    for train_idx, test_idx in gkf.split(x, groups=groups):
        folds.append(
            {
                "train": [object_ids[i] for i in train_idx],
                "test": [object_ids[i] for i in test_idx],
            }
        )
    return folds


def freeze_splits(folds: list[dict[str, list[str]]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"folds": folds}, indent=2))


def load_splits(path: str | Path) -> list[dict[str, list[str]]]:
    return json.loads(Path(path).read_text())["folds"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class EvalRow:
    """One evaluated test object: its truth and the pipeline's selection."""

    object_id: str
    truth: ObjectRecord
    result: SelectionResult


@dataclass
class Metrics:
    force: dict = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    model_recommendation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "force": self.force,
            "feasibility": self.feasibility,
            "selection": self.selection,
            "model_recommendation": self.model_recommendation,
        }


def _force_stats(pairs: list[tuple[float, float]], cfg: Config) -> dict:
    if not pairs:
        return {"n": 0}
    errs = np.array([abs(t - p) for t, p in pairs])
    thr = {
        f"within_{t}n": float(np.mean(errs <= t)) for t in cfg.evaluation.within_thresholds_n
    }
    return {
        "n": len(pairs),
        "mae": float(errs.mean()),
        "rmse": float(np.sqrt((errs**2).mean())),
        "medae": float(np.median(errs)),
        **thr,
    }


def compute_metrics(rows: list[EvalRow], cfg: Config) -> Metrics:
    m = Metrics()

    # --- force + feasibility, per gripper --------------------------------- #
    all_pairs: list[tuple[float, float]] = []
    active_grippers = cfg.prediction.active_grippers
    for gripper in active_grippers:
        pairs: list[tuple[float, float]] = []
        tp = fp = tn = fn = 0
        for row in rows:
            truth = row.truth.get(gripper)
            pred = row.result.candidate_predictions.get(gripper.value)
            if truth is None or pred is None:
                continue
            # feasibility confusion
            if truth.feasible and pred.feasible:
                tp += 1
            elif not truth.feasible and pred.feasible:
                fp += 1
            elif not truth.feasible and not pred.feasible:
                tn += 1
            else:
                fn += 1
            # force error only where both feasible
            if truth.feasible and pred.feasible and truth.min_force_n is not None:
                pairs.append((truth.min_force_n, pred.predicted_normal_force_n))
        m.force[gripper.value] = _force_stats(pairs, cfg)
        all_pairs.extend(pairs)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        m.feasibility[gripper.value] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
        }
    m.force["overall"] = _force_stats(all_pairs, cfg)

    # A one-candidate run evaluates force and feasibility, not gripper choice.
    if len(active_grippers) == 1:
        m.selection = {"applicable": False, "n": 0}
        m.model_recommendation = {"applicable": False, "n": 0}
        return m

    # --- selection + regret ----------------------------------------------- #
    correct = 0
    considered = 0
    infeasible_picks = 0
    regrets: list[float] = []
    for row in rows:
        optimal, oracle_f = row.truth.optimal_grippers()
        if not optimal:  # no feasible gripper in truth -> skip selection scoring
            continue
        considered += 1
        chosen = row.result.desired_gripper
        if chosen in {gripper.value for gripper in optimal}:
            correct += 1
        # regret: what the choice actually costs in true force
        if chosen in ("gecko", "silicone"):
            chosen_rec = row.truth.get(Gripper(chosen))
            if chosen_rec is not None and chosen_rec.feasible and chosen_rec.min_force_n is not None:
                regrets.append(chosen_rec.min_force_n - (oracle_f or 0.0))
            else:
                infeasible_picks += 1
                regrets.append(cfg.force.limit_n - (oracle_f or 0.0))  # penalty
        else:  # chose "none" despite a feasible option existing
            regrets.append(cfg.force.limit_n - (oracle_f or 0.0))
    m.selection = {
        "applicable": True,
        "n": considered,
        "accuracy": (correct / considered) if considered else float("nan"),
        "infeasible_pick_rate": (infeasible_picks / considered) if considered else float("nan"),
        "mean_regret_n": float(statistics.mean(regrets)) if regrets else 0.0,
        "median_regret_n": float(statistics.median(regrets)) if regrets else 0.0,
        "worst_regret_n": float(max(regrets)) if regrets else 0.0,
    }

    # Raw VLM recommendation is scored separately from the authoritative selector.
    recommendation_rows = [
        row for row in rows if row.result.model_recommended_gripper is not None
    ]
    recommendation_correct = 0
    selector_agreement = 0
    for row in recommendation_rows:
        optimal, _ = row.truth.optimal_grippers()
        recommendation = row.result.model_recommended_gripper
        if recommendation in {gripper.value for gripper in optimal}:
            recommendation_correct += 1
        if recommendation == row.result.desired_gripper:
            selector_agreement += 1
    recommendation_count = len(recommendation_rows)
    m.model_recommendation = {
        "applicable": True,
        "n": recommendation_count,
        "accuracy": (
            recommendation_correct / recommendation_count
            if recommendation_count
            else float("nan")
        ),
        "selector_agreement": (
            selector_agreement / recommendation_count
            if recommendation_count
            else float("nan")
        ),
    }
    return m


def records_for_objects(
    records: list[ExperienceRecord], object_ids: list[str]
) -> dict[str, ObjectRecord]:
    wanted = set(object_ids)
    return group_by_object([r for r in records if r.object_id in wanted])
