"""Turn one predicted grasp into the serial lines main.ino executes.

Pure by design: this module never opens a serial port. `SerialGraspSender` in
:mod:`modules.hardware` does the I/O, so the exact bytes sent to the rig can be
pinned by unit tests with no Arduino attached.

See docs/superpowers/specs/2026-07-27-serial-handoff-design.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from .contracts import Gripper

if TYPE_CHECKING:
    from .config import Config
    from .contracts import SelectionResult
    from .datasets.models import DatasetObject

# --------------------------------------------------------------------------- #
# Mirrors of arduino/main/main.ino constants. Keep in sync by hand.
# These are hardware facts, not tunables -- they do not belong in config.yaml.
# --------------------------------------------------------------------------- #
GRIP_MAX_OPENING_MM = 101.6    # main.ino's GRIP_MAX_OPENING_MM
GECKO_SELECT_TOKEN = "GEKKO"   # main.ino's SELECT branch: its spelling of "gecko"

_SELECT_TOKENS = {
    Gripper.GECKO: GECKO_SELECT_TOKEN,
    Gripper.SILICONE: "SILICONE",
}


class GraspCommand(BaseModel):
    """One executable grasp: every value the firmware needs, already validated.

    `gripper` is typed as `Gripper` rather than `GripperChoice` on purpose --
    `GripperChoice` admits "none", and an unrepresentable state beats a
    rejected one.
    """

    object_id: str = Field(min_length=1)
    object_height_mm: float = Field(gt=0)
    object_width_mm: float = Field(gt=0)
    gripper: Gripper
    force_n: float = Field(ge=0)

    @model_validator(mode="after")
    def _width_fits_the_jaws(self) -> GraspCommand:
        # The firmware's response to an over-wide object is silent: main.ino's
        # gripStepsForWidth floors negative travel to zero and leaves the jaws
        # fully open with no WARN. Raising beats a rig that looks like it is
        # working.
        #
        # NOTE: `from_prediction` below duplicates this check so that path
        # never has to surface pydantic's ValidationError (see its docstring).
        # This validator stays so direct `GraspCommand(...)` construction is
        # still guarded.
        if self.object_width_mm > GRIP_MAX_OPENING_MM:
            raise ValueError(
                f"width {self.object_width_mm} mm exceeds jaw opening "
                f"{GRIP_MAX_OPENING_MM} mm"
            )
        return self

    @classmethod
    def from_prediction(
        cls,
        obj: DatasetObject,
        selection: SelectionResult,
        cfg: Config,
    ) -> GraspCommand:
        """Assemble a command from a pipeline selection and an object's contact data.

        Gripper and force come from `Pipeline.predict()` (pipeline.py:33); height
        and width from the contact model's summary.json, surfaced as
        `DatasetObject.contact_fraction` by `datasets/catalog.py`'s
        `_attach_contact_summary`. Those two halves have different lifecycles --
        one live, one read from disk -- which is why they are validated together
        here.

        Raises ValueError naming the fix when the grasp cannot be executed.
        Every rejection here -- including the width and dimension checks that
        `GraspCommand` itself also enforces via `_width_fits_the_jaws` and its
        field constraints -- is a plain `ValueError` raised before `cls(...)`
        is ever called, so a caller catching `ValueError` never has to also
        catch pydantic's `ValidationError` to cover the same failures.
        All-or-nothing: nothing partial reaches a rig that has no homing routine
        and no limit switches.
        """
        object_id = obj.object_id
        if selection.desired_gripper == "none":
            raise ValueError(f"no feasible gripper for {object_id}")
        force_n = selection.predicted_normal_force_n
        if force_n is None:
            raise ValueError(f"selection has no force for {object_id}")
        contact = obj.contact_fraction
        if contact is None:
            raise ValueError(
                f"no contact data for {object_id}: run scripts/calibrate_scale.py to "
                "establish geometry.px_per_mm, then run "
                "prepare_dataset --stages surface_area"
            )
        if not contact.grasp_feasible:
            raise ValueError(f"contact model found no antipodal grasp for {object_id}")
        if contact.object_width_mm > GRIP_MAX_OPENING_MM:
            raise ValueError(
                f"width {contact.object_width_mm} mm exceeds jaw opening "
                f"{GRIP_MAX_OPENING_MM} mm for {object_id}: re-measure or re-run the "
                "contact model"
            )
        if contact.object_height_mm <= 0 or contact.object_width_mm <= 0:
            raise ValueError(
                f"non-positive object dimensions for {object_id} "
                f"(height={contact.object_height_mm} mm, width={contact.object_width_mm} mm): "
                "the contact summary is degenerate, re-run prepare_dataset --stages surface_area"
            )
        if force_n > cfg.force.limit_n:
            raise ValueError(f"force {force_n} N exceeds limit {cfg.force.limit_n} N")
        return cls(
            object_id=object_id,
            object_height_mm=contact.object_height_mm,
            object_width_mm=contact.object_width_mm,
            gripper=Gripper(selection.desired_gripper),
            force_n=force_n,
        )

    def serialize(self) -> list[str]:
        """Render the firmware command lines, in execution order.

        Order is SELECT, then Z, then FORCE, then GRIP CLOSE:

        - SELECT precedes Z so the turret only ever rotates at the TOP of
          travel. The turret is direct-drive with two gripper heads mounted
          96.75 degrees apart (main.ino's SELECT_SILICONE_STEPS geometry),
          hanging 260.35 mm below the carriage. For objects under ~118 mm
          tall the gripper is floored to 10 mm above the floor with the
          object already sitting between the open jaws (main.ino's
          moveZForGrasp) -- rotating the turret there sweeps a head straight
          through the object. There is no limit switch and no torque sensing
          to catch that.
        - Z precedes GRIP CLOSE so the gripper descends before the jaws close
          -- reversed, the jaws close in mid-air and the gripper descends
          already shut.
        - FORCE precedes GRIP CLOSE so the target is set before any jaw
          motion begins.

        PRECONDITION: `object_height_mm` is the object's own vertical extent,
        as measured by the contact model (contact_model/pipeline_core.py's
        surface-area stage), while the firmware's `Z <mm>` is the height of
        the object's TOP above the RIG'S FLOOR. These two are the same number
        only if the object rests directly on the plane the firmware's Z
        geometry references -- confirmed true for this rig. A riser,
        turntable, or scale placed under the object would silently offset
        every grasp downward; nothing in this module or the firmware detects
        that.

        PROTOCOL CONTRACT -- FORCE: `SerialGraspSender._await_ack` requires
        `FORCE <n>` to be acknowledged with `DONE FORCE`, the same
        request/ack shape as `SELECT`, `Z`, and `GRIP`. See
        `SerialGraspSender`'s class docstring.

        Fixed-point formatting is mandatory: main.ino's parseFloatStrict
        accepts only an optional sign, digits, and at most one dot. Python's
        default float formatting emits "1e-05" for small values, which the
        firmware answers with "ERR bad value".
        """
        return [
            f"SELECT {_SELECT_TOKENS[self.gripper]}",
            f"Z {self.object_height_mm:.1f}",
            f"FORCE {self.force_n:.2f}",
            f"GRIP CLOSE {self.object_width_mm:.1f}",
        ]
