"""Public experiment API, re-exported from the modular strategy package."""

from .catalog import EXPERIMENT_CATALOG, create_strategy, experiment_display_name
from .eligibility import (
    EvaluationTruthEligibility,
    ExperimentEligibility,
    evaluation_truth_eligibility,
    experiment_eligibility,
)
from .helper import (
    ExperimentSpec,
    ExperimentStrategy,
    PipelineRunResult,
    QueryInput,
)

__all__ = [
    "EXPERIMENT_CATALOG",
    "EvaluationTruthEligibility",
    "ExperimentEligibility",
    "ExperimentSpec",
    "ExperimentStrategy",
    "PipelineRunResult",
    "QueryInput",
    "create_strategy",
    "evaluation_truth_eligibility",
    "experiment_display_name",
    "experiment_eligibility",
]
