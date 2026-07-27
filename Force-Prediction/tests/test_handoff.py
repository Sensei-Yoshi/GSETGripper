from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.config import load_config
from modules.contracts import Gripper, SelectionResult
from modules.datasets.models import (
    ContactFractionArtifact,
    DatasetObject,
    ImageArtifact,
)
from modules.handoff import GraspCommand

CFG = load_config()


def _command(**overrides) -> GraspCommand:
    values = {
        "object_id": "banana",
        "object_height_mm": 84.2,
        "object_width_mm": 61.0,
        "gripper": Gripper.SILICONE,
        "force_n": 1.75,
    }
    values.update(overrides)
    return GraspCommand(**values)


def test_serialize_emits_exact_firmware_lines():
    assert _command().serialize() == [
        "Z 84.2",
        "SELECT SILICONE",
        "FORCE 1.75",
        "GRIP CLOSE 61.0",
    ]


def test_gecko_serializes_with_the_firmware_spelling():
    # main.ino:223 compares against "GEKKO", not "GECKO". Sending the Python
    # enum value verbatim yields "ERR unknown select position".
    lines = _command(gripper=Gripper.GECKO).serialize()
    assert "SELECT GEKKO" in lines
    assert "SELECT GECKO" not in lines


def test_descends_before_closing_and_sets_force_first():
    lines = _command().serialize()
    z = next(i for i, line in enumerate(lines) if line.startswith("Z "))
    force = next(i for i, line in enumerate(lines) if line.startswith("FORCE "))
    grip = next(i for i, line in enumerate(lines) if line.startswith("GRIP CLOSE "))
    assert z < grip
    assert force < grip


def test_small_force_never_renders_in_scientific_notation():
    # parseFloatStrict (main.ino:340-371) accepts only sign, digits, one dot.
    # "1e-05" comes back as "ERR bad value".
    #
    # Check the ARGUMENT, not the whole line: FORCE, SILICONE, and CLOSE all
    # contain "e", so scanning the line for "e" can never pass.
    lines = _command(force_n=0.00001).serialize()
    force_line = next(line for line in lines if line.startswith("FORCE "))
    assert force_line == "FORCE 0.00"
    assert "e" not in force_line.removeprefix("FORCE ")


def test_rejects_nonpositive_dimensions():
    with pytest.raises(ValidationError):
        _command(object_height_mm=0.0)
    with pytest.raises(ValidationError):
        _command(object_width_mm=-1.0)


def test_rejects_negative_force():
    with pytest.raises(ValidationError):
        _command(force_n=-0.1)


def test_rejects_object_wider_than_the_jaws():
    with pytest.raises(ValueError, match="exceeds jaw opening"):
        _command(object_width_mm=150.0)


def _contact(**overrides) -> ContactFractionArtifact:
    values = {
        "summary_path": "data/expforce/objects/banana/contact_fraction/summary.json",
        "schema_version": 2,
        "object_height_mm": 84.2,
        "object_width_mm": 61.0,
        "geometric_contact_fraction": 0.42,
        "combined_contact_fraction": 0.42,
        "grasp_feasible": True,
        "antipodal_grasp": True,
        "contact_floor_applied": False,
    }
    values.update(overrides)
    return ContactFractionArtifact(**values)


def _object(contact: ContactFractionArtifact | None = None) -> DatasetObject:
    return DatasetObject(
        dataset_id="expforce",
        object_id="banana",
        name="Banana",
        image=ImageArtifact(path="data/expforce/objects/banana/image.png"),
        contact_fraction=contact if contact is not None else _contact(),
    )


def _selection(**overrides) -> SelectionResult:
    values = {"desired_gripper": "silicone", "predicted_normal_force_n": 1.75}
    values.update(overrides)
    return SelectionResult(**values)


def test_from_prediction_assembles_all_four_values():
    cmd = GraspCommand.from_prediction(_object(), _selection(), CFG)
    assert cmd.object_id == "banana"
    assert cmd.object_height_mm == 84.2
    assert cmd.object_width_mm == 61.0
    assert cmd.gripper is Gripper.SILICONE
    assert cmd.force_n == 1.75
    assert cmd.serialize()[0] == "Z 84.2"


def test_from_prediction_rejects_infeasible_selection():
    selection = _selection(desired_gripper="none", predicted_normal_force_n=None)
    with pytest.raises(ValueError, match="no feasible gripper for banana"):
        GraspCommand.from_prediction(_object(), selection, CFG)


def test_from_prediction_rejects_missing_force():
    selection = _selection(predicted_normal_force_n=None)
    with pytest.raises(ValueError, match="selection has no force for banana"):
        GraspCommand.from_prediction(_object(), selection, CFG)


def test_from_prediction_rejects_missing_contact_data():
    obj = DatasetObject(
        dataset_id="expforce",
        object_id="banana",
        name="Banana",
        image=ImageArtifact(path="data/expforce/objects/banana/image.png"),
    )
    with pytest.raises(ValueError, match="run prepare_dataset --stages surface_area"):
        GraspCommand.from_prediction(obj, _selection(), CFG)


def test_from_prediction_rejects_infeasible_contact_geometry():
    obj = _object(_contact(grasp_feasible=False))
    with pytest.raises(ValueError, match="no antipodal grasp for banana"):
        GraspCommand.from_prediction(obj, _selection(), CFG)


def test_from_prediction_rejects_force_over_the_configured_limit():
    selection = _selection(predicted_normal_force_n=CFG.force.limit_n + 0.5)
    with pytest.raises(ValueError, match="exceeds limit"):
        GraspCommand.from_prediction(_object(), selection, CFG)
