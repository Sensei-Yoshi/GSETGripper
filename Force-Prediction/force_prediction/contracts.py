"""Shared data contracts — the single source of truth for the whole pipeline.

Every module imports these models; nothing defines its own ad-hoc dict shape.
This is what lets the team build modules in parallel against stable interfaces.

Records are stored one-per-object-per-gripper in data/experiences.jsonl. The
`ObjectRecord` paired view groups an object's two gripper rows and is what powers
the paired-row retrieval enhancement and the selection oracle.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Gripper(StrEnum):
    GECKO = "gecko"
    SILICONE = "silicone"

    def other(self) -> Gripper:
        return Gripper.SILICONE if self is Gripper.GECKO else Gripper.GECKO


class Compatibility(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Meta(BaseModel):
    """Provenance for one measured object-gripper pair (kept for auditability)."""

    temp_c: float | None = None
    humidity_pct: float | None = None
    pad_id: str | None = None
    trial_forces_n: list[float] = Field(default_factory=list)
    n_trials: int = 0
    date: str | None = None
    curvature_tag: str | None = None  # future geometry work (backlog)


class ExperienceRecord(BaseModel):
    """One measured object-gripper pair (one row of experiences.jsonl)."""

    object_id: str
    image_path: str
    mass_g: float = Field(gt=0)
    roughness_class: int = Field(ge=1, le=5)
    projected_contact_fraction: float = Field(ge=0, le=1)
    gripper: Gripper
    min_force_n: float | None = None
    feasible: bool = True
    failed_at_limit_n: float | None = None
    semantic_description: str = ""
    meta: Meta = Field(default_factory=Meta)

    @model_validator(mode="after")
    def _feasibility_consistency(self) -> ExperienceRecord:
        # Feasible: min_force_n is the real minimum; failed_at_limit_n unused.
        # Infeasible: min_force_n is None and we record the limit we failed at,
        # so "failed at 8 N" is never confused with "minimum is 8 N".
        if self.feasible:
            if self.min_force_n is None:
                raise ValueError("feasible record requires min_force_n")
            if self.min_force_n <= 0:
                raise ValueError("min_force_n must be positive")
        else:
            if self.min_force_n is not None:
                raise ValueError("infeasible record must have min_force_n=None")
            if self.failed_at_limit_n is None:
                raise ValueError("infeasible record requires failed_at_limit_n")
        return self


class ObjectRecord(BaseModel):
    """Paired view of one object's two gripper rows.

    Either side may be None (only one gripper tested). `other_gripper_force`
    supplies the cross-material force used by the paired-row enhancement and the
    selection oracle.
    """

    object_id: str
    gecko: ExperienceRecord | None = None
    silicone: ExperienceRecord | None = None

    def get(self, gripper: Gripper) -> ExperienceRecord | None:
        return self.gecko if gripper is Gripper.GECKO else self.silicone

    def other_gripper_force(self, gripper: Gripper) -> float | None:
        """Feasible min force of the *other* gripper for this same object."""
        rec = self.get(gripper.other())
        return rec.min_force_n if (rec is not None and rec.feasible) else None

    def oracle(self) -> tuple[Gripper | None, float | None]:
        """Best (lowest-force feasible) gripper and its force; (None, None) if neither feasible."""
        candidates = [
            (g, rec.min_force_n)
            for g, rec in ((Gripper.GECKO, self.gecko), (Gripper.SILICONE, self.silicone))
            if rec is not None and rec.feasible and rec.min_force_n is not None
        ]
        if not candidates:
            return None, None
        return min(candidates, key=lambda gf: gf[1])

    def optimal_grippers(self) -> tuple[set[Gripper], float | None]:
        """Return every minimum-force gripper so tied labels score fairly."""
        candidates = {
            g: rec.min_force_n
            for g, rec in ((Gripper.GECKO, self.gecko), (Gripper.SILICONE, self.silicone))
            if rec is not None and rec.feasible and rec.min_force_n is not None
        }
        if not candidates:
            return set(), None
        minimum = min(candidates.values())
        return {g for g, force in candidates.items() if force == minimum}, minimum


class Query(BaseModel):
    """Measured properties of a query object (no force labels)."""

    object_id: str
    image_path: str
    mass_g: float = Field(gt=0)
    roughness_class: int = Field(ge=1, le=5)
    projected_contact_fraction: float = Field(ge=0, le=1)
    semantic_description: str = ""


class CandidateQuery(Query):
    """A query bound to one candidate gripper for a per-gripper prediction."""

    candidate_gripper: Gripper


class PerGripperPrediction(BaseModel):
    """VLM (or non-VLM) prediction for a single candidate gripper."""

    candidate_gripper: Gripper
    visible_surface_material: str = "unknown"
    visible_surface_condition: str = "unknown"
    compatibility: Compatibility = Compatibility.UNKNOWN
    feasible: bool = True
    predicted_normal_force_n: float = Field(ge=0)
    reasoning_trace: str = ""


class SelectionResult(BaseModel):
    """Final deterministic selection across candidate grippers."""

    desired_gripper: str  # "gecko" | "silicone" | "none"
    predicted_normal_force_n: float | None = None
    candidate_predictions: dict[str, PerGripperPrediction] = Field(default_factory=dict)
    reasoning_trace: str = ""


# --------------------------------------------------------------------------- #
# JSONL IO + paired grouping
# --------------------------------------------------------------------------- #

def load_experiences(path: str | Path) -> list[ExperienceRecord]:
    path = Path(path)
    records: list[ExperienceRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ExperienceRecord.model_validate_json(line))
    return records


def save_experiences(path: str | Path, records: list[ExperienceRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def append_experience(path: str | Path, record: ExperienceRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def group_by_object(records: list[ExperienceRecord]) -> dict[str, ObjectRecord]:
    """Group flat rows into paired ObjectRecords keyed by object_id."""
    objects: dict[str, ObjectRecord] = {}
    for rec in records:
        obj = objects.setdefault(rec.object_id, ObjectRecord(object_id=rec.object_id))
        if rec.gripper is Gripper.GECKO:
            obj.gecko = rec
        else:
            obj.silicone = rec
    return objects
