"""Hardware abstraction: interfaces + real serial drivers + physics-backed mocks.

Everything physical sits behind a `Protocol` so the collection controller and the
whole ML pipeline run headless on mocks (no serial, no camera, no API). Real
drivers reuse the serial-autodetect + newline-command pattern already used in
GSETGripper/camera/depth_serial_trigger.py.

Interfaces
    GripperController : set_normal_force / open / close_until_contact / attempt_lift
    LoadCell          : read_n
    RoughnessSource   : read_class          (the trusted LED system)
    MassSource        : read_g              (the scale)
    CameraSource      : capture_rgb
    SerialGraspSender : send                (arduino/main/main.ino's Z/SELECT/FORCE/GRIP
                                              protocol -- see its own docstring)

`SerialGripper` below targets a SET_FORCE <n> / OPEN / CLOSE / READ / LIFT
protocol over serial at 9600 baud -- but that firmware, firmware/gripper_force,
has never been written; see `SerialGripper`'s own docstring. The firmware that
does exist, arduino/main/main.ino, speaks the unrelated Z/SELECT/FORCE/GRIP
protocol instead, and is driven by `SerialGraspSender`, not `SerialGripper`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .config import Config
from .contracts import ExperienceRecord, Gripper, Meta
from .physics import PhysicsModel, PhysicsParams, weight_n

if TYPE_CHECKING:
    from .handoff import GraspCommand


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
@runtime_checkable
class GripperController(Protocol):
    def set_normal_force(self, newtons: float) -> None: ...
    def close_until_contact(self) -> None: ...
    def open(self) -> None: ...
    def attempt_lift(self) -> bool:
        """Perform the standardized lift; return True iff the object is held."""
        ...


@runtime_checkable
class LoadCell(Protocol):
    def read_n(self) -> float: ...


@runtime_checkable
class RoughnessSource(Protocol):
    def read_class(self) -> int: ...


@runtime_checkable
class MassSource(Protocol):
    def read_g(self) -> float: ...


@runtime_checkable
class CameraSource(Protocol):
    def capture_rgb(self) -> np.ndarray: ...


@dataclass
class Bench:
    """Bundle of the hardware a collection run needs."""

    gripper: GripperController
    load_cell: LoadCell
    roughness: RoughnessSource
    mass: MassSource
    camera: CameraSource


# --------------------------------------------------------------------------- #
# Real serial helpers / drivers  (imported lazily; not needed for mock runs)
# --------------------------------------------------------------------------- #
def find_serial_port(preferred: str | None = None) -> str:
    """Autodetect an Arduino-like serial port (mirrors depth_serial_trigger)."""
    from serial.tools import list_ports

    if preferred:
        return preferred
    candidates = []
    for port in list_ports.comports():
        desc = f"{port.description} {port.manufacturer or ''}".lower()
        dev = port.device.lower()
        if any(t in desc for t in ("arduino", "ch340", "usb serial")) or any(
            t in dev for t in ("usbmodem", "usbserial")
        ):
            candidates.append(port.device)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple serial ports found; pass one explicitly: {candidates}")
    available = ", ".join(p.device for p in list_ports.comports()) or "none"
    raise RuntimeError(f"No Arduino serial port found. Available: {available}")


class SerialGripper:
    """Real gripper + load cell over one Arduino link (firmware/gripper_force).

    NOTE: this targets firmware that has NOT been written. `arduino/main/main.ino`
    speaks a different protocol (Z / SELECT / GRIP -- see `SerialGraspSender`
    below) and implements none of the commands used here.

    Kept because it is the only written record of what main.ino must still grow
    before `collect_real` can run on real hardware:
        SET_FORCE <n>   set the stationary-finger target force
        CLOSE           close until contact
        OPEN            release
        READ            report the load-cell reading in newtons
        LIFT            perform the standardized lift; reply HELD or not
    """

    def __init__(self, cfg: Config, port: str | None = None, baud: int = 9600) -> None:
        import serial

        self.cfg = cfg
        self.port = find_serial_port(port)
        self.conn = serial.Serial(self.port, baud, timeout=2)
        time.sleep(2.0)  # allow the board to reset

    def _cmd(self, line: str) -> str:
        self.conn.write((line.strip() + "\n").encode("ascii"))
        self.conn.flush()
        return self.conn.readline().decode("ascii", errors="ignore").strip()

    def set_normal_force(self, newtons: float) -> None:
        self._cmd(f"SET_FORCE {newtons:.3f}")

    def close_until_contact(self) -> None:
        self._cmd("CLOSE")

    def open(self) -> None:
        self._cmd("OPEN")

    def attempt_lift(self) -> bool:
        return self._cmd("LIFT").upper().startswith("HELD")

    def read_n(self) -> float:
        reply = self._cmd("READ")
        try:
            return float(reply)
        except ValueError:
            return float("nan")

    def close(self) -> None:
        self.conn.close()


class SerialRoughness:
    """Real LED roughness system: returns an integer class over serial."""

    def __init__(self, port: str | None = None, baud: int = 9600) -> None:
        import serial

        self.conn = serial.Serial(find_serial_port(port), baud, timeout=2)
        time.sleep(2.0)

    def read_class(self) -> int:
        self.conn.write(b"READ\n")
        self.conn.flush()
        return int(self.conn.readline().decode("ascii", errors="ignore").strip())


# Command word -> the DONE ack main.ino emits once that axis's move completes.
# Not one-to-one with GraspCommand.serialize()'s lines: both "GRIP CLOSE <mm>"
# and "GRIP OPEN" ack as "DONE GRIP".
_EXPECTED_DONE = {
    "Z": "DONE Z",
    "SELECT": "DONE SELECT",
    "GRIP": "DONE GRIP",
    "FORCE": "DONE FORCE",
}


class SerialGraspSender:
    """Send one GraspCommand to arduino/main/main.ino, one line at a time.

    Writes a line, waits for its DONE ack, then writes the next -- rather than
    dumping all four. The firmware sets a pending flag and prints DONE <axis>
    only once distanceToGo() reaches 0 (main.ino's loop(), the pending-flag
    block), so waiting is what keeps the moves sequential instead of
    overlapping.

    PHYSICAL PRECONDITION: opening the port resets the board, and the firmware
    then ASSUMES the carriage is parked at the top of travel and the jaws are
    fully open. There is no homing routine and no limit switch (see main.ino's
    Z-geometry and GRIP-geometry comment blocks). Park the rig before
    constructing this, or the gripper is driven into the table with nothing in
    hardware to stop it.

    POSTCONDITION -- send() DOES NOT PARK THE RIG: a successful send() leaves
    the carriage at grasp depth with the jaws closed around the object. It
    does not retract, and there is no `park()` method. Because opening the
    port resets the board and the firmware then assumes, on that reset,
    carriage-at-top-of-travel and jaws-fully-open, the operator MUST MANUALLY
    PARK THE RIG before the *next* connection -- otherwise the firmware's
    position origin is false, and the next `Z` command drives the gripper
    down from a wrong reference, through the table. (main.ino clamps the Z
    argument to its reachable band, so manually sending `Z 360.68` -- or
    anything at or above it -- is a way to drive the carriage back to the top
    of travel as part of parking.)

    PROTOCOL CONTRACT -- FORCE: `_await_ack` requires `FORCE <n>` to be
    acknowledged with `DONE FORCE`, the same request/ack shape as `SELECT`,
    `Z`, and `GRIP CLOSE <mm>` / `GRIP OPEN`.
    """

    def __init__(
        self,
        port: str | None = None,
        baud: int = 9600,
        timeout: float = 30.0,
    ) -> None:
        import serial

        self.port = find_serial_port(port)
        self.conn = serial.Serial(self.port, baud, timeout=timeout)
        time.sleep(2.0)  # allow the board to reset

    def send(self, cmd: GraspCommand) -> list[str]:
        """Execute the grasp. Returns any WARN lines the firmware emitted.

        A WARN means the firmware clamped something -- the rig did not do
        exactly what was asked, so callers should surface these.
        """
        warnings: list[str] = []
        for line in cmd.serialize():
            self.conn.write((line + "\n").encode("ascii"))
            self.conn.flush()
            warnings.extend(self._await_ack(line))
        return warnings

    def _await_ack(self, line: str) -> list[str]:
        """Wait for the specific DONE ack that acknowledges `line`.

        Every connection resets the board, so boot-time garbage or a
        desynchronised stream is a real scenario -- accepting *any* line
        starting with "DONE" would let a stale ack for a different axis pass
        as success. The mapping from command word to ack is uniform but not
        one-to-one: `Z` -> `DONE Z`, `SELECT ...` -> `DONE SELECT`,
        `FORCE ...` -> `DONE FORCE`, `GRIP CLOSE ...` (and `GRIP OPEN`) ->
        `DONE GRIP`.
        """
        expected = _EXPECTED_DONE[line.split(" ", 1)[0]]
        warnings: list[str] = []
        while True:
            reply = self.conn.readline().decode("ascii", errors="ignore").strip()
            if not reply:
                raise TimeoutError(f"no reply from firmware for {line!r}")
            if reply.startswith("ERR"):
                # processCommand returns after sendErr without setting a pending
                # flag (main.ino's sendErr), so no DONE will ever follow. Return
                # now; waiting for one blocks forever.
                raise RuntimeError(f"firmware rejected {line!r}: {reply}")
            if reply.startswith("WARN"):
                warnings.append(reply)
                continue
            if reply == expected:
                return warnings
            if reply.startswith("DONE"):
                raise RuntimeError(
                    f"firmware desynchronised: expected {expected!r} "
                    f"for {line!r}, got {reply!r}"
                )
            raise RuntimeError(f"unexpected firmware reply to {line!r}: {reply}")

    def close(self) -> None:
        self.conn.close()


class ManualMass:
    """Fallback mass source: operator types the scale reading."""

    def read_g(self) -> float:
        return float(input("Enter measured mass in grams: ").strip())


class OrbbecCamera:
    """Real Astra+ capture. Lazily imports the SDK so mock runs need no build."""

    def __init__(self) -> None:
        from pyorbbecsdk import Config as OBConfig
        from pyorbbecsdk import OBSensorType, Pipeline

        self.pipeline = Pipeline()
        conf = OBConfig()
        color = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        conf.enable_stream(color.get_default_video_stream_profile())
        self.pipeline.start(conf)

    def capture_rgb(self) -> np.ndarray:
        import cv2

        frames = self.pipeline.wait_for_frames(500)
        color = frames.get_color_frame()
        w, h = color.get_width(), color.get_height()
        buf = np.frombuffer(color.get_data(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size != w * h * 3 else buf.reshape(h, w, 3)


# --------------------------------------------------------------------------- #
# Mocks — a shared synthetic bench so the whole stack runs with no hardware.
# --------------------------------------------------------------------------- #
@dataclass
class MockObject:
    object_id: str
    mass_g: float
    roughness_class: int
    projected_contact_fraction: float
    # Per-object perturbation of the "true" coefficients so the mean-physics
    # prior is imperfect -> retrieval and residual learning have real signal.
    alpha_scale: float = 1.0
    beta_scale: float = 1.0


@dataclass
class MockBench:
    """Ground-truth simulator shared by all mock devices."""

    cfg: Config
    rng: np.random.Generator
    current: MockObject | None = None
    mounted_gripper: Gripper = Gripper.SILICONE  # collect.py sets per pad
    _commanded_n: float = 0.0
    _base: PhysicsParams = field(init=False)

    def __post_init__(self) -> None:
        self._base = PhysicsParams.from_config(self.cfg)

    def set_object(self, obj: MockObject) -> None:
        self.current = obj
        self._commanded_n = 0.0

    def _true_model(self, obj: MockObject) -> PhysicsModel:
        p = PhysicsParams(**vars(self._base))
        p.alpha_sil0 *= obj.alpha_scale
        p.alpha_geo0 *= obj.alpha_scale
        p.beta0 *= obj.beta_scale
        return PhysicsModel(p, self.cfg)

    def lift_succeeds(self, gripper: Gripper, trial: int) -> bool:
        assert self.current is not None
        obj = self.current
        model = self._true_model(obj)
        w = weight_n(obj.mass_g, self.cfg.force.gravity)
        held = model.holding_force(
            gripper, self._commanded_n, obj.roughness_class, obj.projected_contact_fraction
        )
        # Small per-trial threshold noise so 3 repeats can differ slightly.
        noise = 1.0 + 0.03 * self.rng.standard_normal()
        return held >= w * noise


class MockGripper:
    def __init__(self, bench: MockBench) -> None:
        self.bench = bench
        self._trial = 0

    def set_normal_force(self, newtons: float) -> None:
        self.bench._commanded_n = newtons

    def close_until_contact(self) -> None:
        self.bench._commanded_n = self.bench.cfg.force.min_n

    def open(self) -> None:
        self.bench._commanded_n = 0.0

    def attempt_lift(self) -> bool:
        # gripper identity is derived from which pad is mounted; collect.py sets it
        self._trial += 1
        return self.bench.lift_succeeds(self.bench.mounted_gripper, self._trial)


class MockLoadCell:
    def __init__(self, bench: MockBench) -> None:
        self.bench = bench

    def read_n(self) -> float:
        return self.bench._commanded_n + 0.01 * self.bench.rng.standard_normal()


class MockRoughness:
    def __init__(self, bench: MockBench) -> None:
        self.bench = bench

    def read_class(self) -> int:
        assert self.bench.current is not None
        return self.bench.current.roughness_class


class MockMass:
    def __init__(self, bench: MockBench) -> None:
        self.bench = bench

    def read_g(self) -> float:
        assert self.bench.current is not None
        return self.bench.current.mass_g


class MockCamera:
    """Generates a deterministic synthetic RGB image per object."""

    def __init__(self, bench: MockBench) -> None:
        self.bench = bench

    def capture_rgb(self) -> np.ndarray:
        import cv2

        assert self.bench.current is not None
        obj = self.bench.current
        h = int(abs(hash(obj.object_id)) % 2**32)
        color = ((h & 0xFF), (h >> 8 & 0xFF), (h >> 16 & 0xFF))
        img = np.full((256, 256, 3), color, dtype=np.uint8)
        cv2.putText(img, obj.object_id, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        return img

def make_mock_bench(cfg: Config, seed: int | None = None) -> tuple[MockBench, Bench]:
    """Build a mock bench + wired-up devices."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    bench = MockBench(cfg=cfg, rng=rng)
    devices = Bench(
        gripper=MockGripper(bench),
        load_cell=MockLoadCell(bench),
        roughness=MockRoughness(bench),
        mass=MockMass(bench),
        camera=MockCamera(bench),
    )
    return bench, devices


def fabricate_records(cfg: Config, n: int, seed: int | None = None) -> list[ExperienceRecord]:
    """Build an in-memory paired dataset (both grippers per object) using the
    perturbed 'true' physics. Used by tests and the check_* scripts so they run
    with zero setup; mirrors what collect --mock writes (minus staircase noise)."""
    bench, _ = make_mock_bench(cfg, seed=seed)
    records: list[ExperienceRecord] = []
    for obj in synthetic_objects(cfg, n, seed=seed):
        bench.set_object(obj)
        for gripper in (Gripper.GECKO, Gripper.SILICONE):
            est = bench._true_model(obj).min_force(
                gripper, obj.mass_g, obj.roughness_class, obj.projected_contact_fraction
            )
            records.append(
                ExperienceRecord(
                    object_id=obj.object_id,
                    image_path="",
                    mass_g=obj.mass_g,
                    roughness_class=obj.roughness_class,
                    projected_contact_fraction=obj.projected_contact_fraction,
                    gripper=gripper,
                    min_force_n=est.min_force_n if est.feasible else None,
                    feasible=est.feasible,
                    failed_at_limit_n=None if est.feasible else cfg.force.limit_n,
                    semantic_description=f"synthetic object {obj.object_id}",
                    meta=Meta(n_trials=0, pad_id="fabricated"),
                )
            )
    return records


def synthetic_objects(cfg: Config, n: int, seed: int | None = None) -> list[MockObject]:
    """Sample a spread of synthetic objects covering mass/roughness/contact space."""
    rng = np.random.default_rng((cfg.seed + 1) if seed is None else seed)
    objects = []
    for i in range(n):
        mass = float(np.exp(rng.uniform(np.log(20), np.log(1500))))  # 20 g .. 1.5 kg
        roughness = int(rng.integers(1, cfg.roughness.n_classes + 1))
        contact = float(rng.uniform(0.3, 1.0))
        objects.append(
            MockObject(
                object_id=f"object_{i:03d}",
                mass_g=round(mass, 1),
                roughness_class=roughness,
                projected_contact_fraction=round(contact, 3),
                alpha_scale=float(np.exp(0.15 * rng.standard_normal())),
                beta_scale=float(np.exp(0.20 * rng.standard_normal())),
            )
        )
    return objects
