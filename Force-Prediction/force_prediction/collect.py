"""Ground-truth collection: coarse-to-fine minimum-force staircase.

Runs against real hardware or the physics-backed mock bench through the same
`Bench` interface, so collection and ML work proceed in parallel (mock unblocks
everything before the gripper firmware exists).

Per object-gripper pair: bracket the minimum lift force with the configured
coarse search step, refine with the configured fine search step, repeat the
configured number of times, and take the median. These are measurement-protocol
settings, not hardware command or model-output quantization. Infeasible (never
lifts within the safe limit) is recorded honestly as feasible=False with
failed_at_limit_n set.

    python -m force_prediction.collect --mock --n 30
    python -m force_prediction.collect --port /dev/cu.usbmodem1101   # real
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
from pathlib import Path

from .config import Config, load_config
from .contracts import ExperienceRecord, Gripper, Meta, append_experience
from .hardware import Bench, MockObject, make_mock_bench, synthetic_objects
from .perception import describe, projected_contact_fraction

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


def measure_min_force_once(devices: Bench, cfg: Config, coarse: float, fine: float) -> float | None:
    """One staircase measurement. Returns the min successful force, or None if the
    object never lifts within the safe limit."""
    limit, start = cfg.force.limit_n, cfg.collection.start_n

    def lifts_at(n: float) -> bool:
        devices.gripper.close_until_contact()
        devices.gripper.set_normal_force(n)
        held = devices.gripper.attempt_lift()
        devices.gripper.open()
        return held

    # --- coarse bracket --- #
    n = start
    bracket: float | None = None
    while n <= limit + 1e-9:
        if lifts_at(round(n, 6)):
            bracket = round(n, 6)
            break
        n = round(n + coarse, 6)
    if bracket is None:
        return None

    # --- fine refine within (bracket - coarse, bracket] --- #
    lo = max(start, round(bracket - coarse, 6))
    n = lo
    while n < bracket - 1e-9:
        if lifts_at(round(n, 6)):
            return round(n, 6)
        n = round(n + fine, 6)
    return bracket


def measure_pair(
    devices: Bench, cfg: Config, repeats: int, coarse: float, fine: float
) -> tuple[bool, float | None, list[float]]:
    """Repeat the staircase; majority-feasible -> median, else infeasible."""
    successes = [
        f for _ in range(repeats)
        if (f := measure_min_force_once(devices, cfg, coarse, fine)) is not None
    ]
    if len(successes) >= (repeats // 2 + 1):
        return True, float(statistics.median(successes)), successes
    return False, None, successes


def _build_record(
    cfg: Config,
    object_id: str,
    image_path: str,
    mass_g: float,
    roughness_class: int,
    contact: float,
    gripper: Gripper,
    description: str,
    feasible: bool,
    min_force: float | None,
    trials: list[float],
    pad_id: str | None,
) -> ExperienceRecord:
    return ExperienceRecord(
        object_id=object_id,
        image_path=image_path,
        mass_g=mass_g,
        roughness_class=roughness_class,
        projected_contact_fraction=contact,
        gripper=gripper,
        min_force_n=min_force if feasible else None,
        feasible=feasible,
        failed_at_limit_n=None if feasible else cfg.force.limit_n,
        semantic_description=description,
        meta=Meta(
            trial_forces_n=trials,
            n_trials=len(trials),
            date=dt.date.today().isoformat(),
            pad_id=pad_id,
        ),
    )


def collect_mock(cfg: Config, n: int, out_path: Path, coarse: float, fine: float) -> None:
    import cv2

    bench, devices = make_mock_bench(cfg)
    images_dir = cfg.path("images")
    images_dir.mkdir(parents=True, exist_ok=True)
    objects: list[MockObject] = synthetic_objects(cfg, n)

    for obj in objects:
        bench.set_object(obj)
        # capture + describe once per object (shared image across grippers)
        rgb = devices.camera.capture_rgb()
        depth = devices.camera.capture_depth_mm()
        image_rel = f"{cfg.paths.images}/{obj.object_id}.png"
        cv2.imwrite(str(cfg.root / image_rel), rgb)
        contact = projected_contact_fraction(depth, cfg)  # perception recovers a
        description = describe(rgb, cfg).description
        mass_g = devices.mass.read_g()
        roughness = devices.roughness.read_class()

        summary = []
        for gripper in GRIPPERS:
            bench.mounted_gripper = gripper
            feasible, min_force, trials = measure_pair(
                devices, cfg, cfg.collection.repeats, coarse, fine
            )
            record = _build_record(
                cfg, obj.object_id, image_rel, mass_g, roughness, contact, gripper,
                description, feasible, min_force, trials, pad_id="mock",
            )
            append_experience(out_path, record)
            summary.append(f"{gripper.value}={'infeasible' if not feasible else f'{min_force}N'}")
        print(f"{obj.object_id}: mass={mass_g:.0f}g rough={roughness} a={contact:.2f}  " + " ".join(summary))
    print(f"Wrote {2 * len(objects)} records to {out_path}")


def collect_real(cfg: Config, out_path: Path, coarse: float, fine: float, port: str | None) -> None:
    """Interactive real-hardware collection. Prompts the operator to place objects
    and swap pads; measures both grippers per object. Requires the gripper firmware,
    the LED roughness system, a scale, and the Astra+ camera."""
    import cv2

    from .hardware import ManualMass, OrbbecCamera, SerialGripper, SerialRoughness

    gripper_dev = SerialGripper(cfg, port=port)
    devices = Bench(
        gripper=gripper_dev,
        load_cell=gripper_dev,
        roughness=SerialRoughness(),
        mass=ManualMass(),
        camera=OrbbecCamera(),
    )
    images_dir = cfg.path("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    while True:
        object_id = input("Object id (blank to finish): ").strip()
        if not object_id:
            break
        input("Place object at the centered grasp location, then press Enter...")
        rgb = devices.camera.capture_rgb()
        depth = devices.camera.capture_depth_mm()
        image_rel = f"{cfg.paths.images}/{object_id}.png"
        cv2.imwrite(str(cfg.root / image_rel), rgb)
        contact = projected_contact_fraction(depth, cfg)
        description = describe(rgb, cfg).description
        mass_g = devices.mass.read_g()
        roughness = devices.roughness.read_class()

        for gripper in GRIPPERS:
            input(f"Mount the {gripper.value.upper()} pad, clean it, then press Enter...")
            pad_id = input(f"{gripper.value} pad id: ").strip() or None
            feasible, min_force, trials = measure_pair(
                devices, cfg, cfg.collection.repeats, coarse, fine
            )
            record = _build_record(
                cfg, object_id, image_rel, mass_g, roughness, contact, gripper,
                description, feasible, min_force, trials, pad_id,
            )
            append_experience(out_path, record)
            print(f"  {gripper.value}: {'INFEASIBLE' if not feasible else f'{min_force} N'}")
    print(f"Appended records to {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ground-truth minimum-force data.")
    p.add_argument("--mock", action="store_true", help="Use the physics-backed mock bench.")
    p.add_argument("--n", type=int, default=30, help="Number of synthetic objects (mock only).")
    p.add_argument("--port", default=None, help="Gripper serial port (real mode).")
    p.add_argument("--out", default=None, help="Output JSONL (defaults to config paths.experiences).")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip live VLM description calls (implied by --mock).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    if args.mock or args.dry_run:
        cfg.models.dry_run = True  # mock collection is fully offline
    out_path = Path(args.out) if args.out else cfg.path("experiences")
    coarse = cfg.collection.coarse_step_n
    fine = cfg.collection.fine_step_n
    if args.mock:
        collect_mock(cfg, args.n, out_path, coarse, fine)
    else:
        collect_real(cfg, out_path, coarse, fine, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
