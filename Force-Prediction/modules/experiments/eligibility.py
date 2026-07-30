"""Experiment-specific readiness for partial datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import Config

if TYPE_CHECKING:
    from ..datasets.models import Dataset, DatasetObject


@dataclass(frozen=True)
class ExperimentEligibility:
    experiment_id: str
    query_ids: tuple[str, ...]
    benchmark_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    skipped_queries: dict[str, tuple[str, ...]]
    skipped_benchmarks: dict[str, tuple[str, ...]]

    def query_reasons(self, object_id: str) -> tuple[str, ...]:
        return self.skipped_queries.get(object_id, ())


@dataclass(frozen=True)
class EvaluationTruthEligibility:
    """Truth readiness for an already-generated prediction batch."""

    eligible_ids: tuple[str, ...]
    skipped: dict[str, tuple[str, ...]]


def _input_reasons(item: DatasetObject, cfg: Config, experiment_id: str) -> list[str]:
    reasons: list[str] = []
    if not item.image.available or not (cfg.root / item.image.path).is_file():
        reasons.append("image unavailable")
    if experiment_id in {"e2", "e4"}:
        if item.mass_g is None:
            reasons.append("mass not recorded")
        if cfg.inputs.use_roughness and item.roughness_index is None:
            reasons.append("roughness index not recorded")
        if (
            cfg.inputs.use_projected_contact
            and item.projected_contact_fraction is None
        ):
            reasons.append("projected contact fraction not recorded")
    return reasons


def _reference_ready(item: DatasetObject, cfg: Config, experiment_id: str) -> bool:
    if not all(
        (outcome := item.gripper_outcomes.get(gripper)) is not None
        and outcome.complete
        for gripper in cfg.prediction.active_grippers
    ):
        return False
    if experiment_id == "e4":
        if item.mass_g is None:
            return False
        if cfg.inputs.use_roughness and item.roughness_index is None:
            return False
        if cfg.inputs.use_projected_contact and item.projected_contact_fraction is None:
            return False
    return True


def _truth_reasons(item: DatasetObject, cfg: Config) -> list[str]:
    active = cfg.prediction.active_grippers
    outcomes = [item.gripper_outcomes.get(gripper) for gripper in active]
    if any(outcome is None or not outcome.complete for outcome in outcomes):
        names = ", ".join(gripper.value for gripper in active)
        return [f"complete truth not recorded for active grippers: {names}"]
    if len(active) == 1:
        return []
    candidates = [
        outcome.min_force_n
        for outcome in outcomes
        if outcome is not None and outcome.feasible and outcome.min_force_n is not None
    ]
    if not candidates:
        return ["neither gripper has a feasible truth label"]
    if len(candidates) == 2 and candidates[0] == candidates[1]:
        return ["paired truth does not have a strict winner"]
    return []


def evaluation_truth_eligibility(
    dataset: Dataset,
    cfg: Config,
    object_ids: tuple[str, ...] | list[str],
) -> EvaluationTruthEligibility:
    """Return truth readiness without rechecking historical generation inputs."""
    eligible: list[str] = []
    skipped: dict[str, tuple[str, ...]] = {}
    for object_id in object_ids:
        item = dataset.objects.get(object_id)
        if item is None:
            skipped[object_id] = ("object no longer exists in the dataset",)
            continue
        reasons = _truth_reasons(item, cfg)
        if reasons:
            skipped[object_id] = tuple(reasons)
        else:
            eligible.append(object_id)
    return EvaluationTruthEligibility(
        eligible_ids=tuple(eligible),
        skipped=skipped,
    )


def experiment_eligibility(
    dataset: Dataset,
    cfg: Config,
    experiment_id: str,
) -> ExperimentEligibility:
    """Return query/reference/benchmark readiness without requiring full coverage."""
    experiment = experiment_id.lower()
    if experiment not in {"e1", "e2", "e3", "e4"}:
        raise KeyError(f"unknown active experiment {experiment_id!r}")

    reference_ids = tuple(
        sorted(
            item.object_id
            for item in dataset.objects.values()
            if experiment in {"e3", "e4"}
            and item.split == "train"
            and _reference_ready(item, cfg, experiment)
        )
    )
    skipped_queries: dict[str, tuple[str, ...]] = {}
    skipped_benchmarks: dict[str, tuple[str, ...]] = {}
    query_ids: list[str] = []
    benchmark_ids: list[str] = []
    for item in dataset.objects.values():
        reasons = _input_reasons(item, cfg, experiment)
        if experiment in {"e3", "e4"} and not any(
            dataset.objects[reference_id].surface_id != item.surface_id
            for reference_id in reference_ids
        ):
            reasons.append("no eligible reference object remains after query exclusion")
        if reasons:
            skipped_queries[item.object_id] = tuple(reasons)
        else:
            query_ids.append(item.object_id)

        benchmark_reasons = [*reasons, *_truth_reasons(item, cfg)]
        if benchmark_reasons:
            skipped_benchmarks[item.object_id] = tuple(dict.fromkeys(benchmark_reasons))
        else:
            benchmark_ids.append(item.object_id)

    return ExperimentEligibility(
        experiment_id=experiment,
        query_ids=tuple(sorted(query_ids)),
        benchmark_ids=tuple(sorted(benchmark_ids)),
        reference_ids=reference_ids,
        skipped_queries=skipped_queries,
        skipped_benchmarks=skipped_benchmarks,
    )
