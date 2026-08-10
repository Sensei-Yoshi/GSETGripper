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

    python -m modules.collect --mock --dataset mock --n 30
    python -m modules.collect --dataset collected --port /dev/cu.usbmodem1101
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
import tempfile
from pathlib import Path

from .config import Config, load_config
from .contact_model import ContactParams, analyze_image, create_rembg_session
from .contracts import ExperienceRecord, Gripper, Meta, append_experience
from .datasets.catalog import slug
from .hardware import Bench, MockObject, make_mock_bench, synthetic_objects
from .perception import describe

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
    roughness_index: float,
    contact: float,
    gripper: Gripper,
    description: str,
    feasible: bool,
    min_force: float | None,
    trials: list[float],
    pad_id: str | None,
    *,
    contact_source: str,
    contact_schema_version: int | None = None,
    contact_summary_path: str | None = None,
    contact_grasp_feasible: bool | None = None,
    contact_antipodal_grasp: bool | None = None,
    contact_floor_applied: bool | None = None,
) -> ExperienceRecord:
    return ExperienceRecord(
        object_id=object_id,
        image_path=image_path,
        mass_g=mass_g,
        roughness_index=roughness_index,
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
            contact_fraction_source=contact_source,
            contact_model_schema_version=contact_schema_version,
            contact_summary_path=contact_summary_path,
            contact_grasp_feasible=contact_grasp_feasible,
            contact_antipodal_grasp=contact_antipodal_grasp,
            contact_floor_applied=contact_floor_applied,
        ),
    )


DATASET_HEADER = [
    "Object",
    "Image",
    "Mass_g",
    "roughness_index",
    "projected_contact_fraction",
    "silicone_force_n",
    "silicone_feasible",
    "gecko_force_n",
    "gecko_feasible",
    "favored_gripper",
]


def _dataset_root(cfg: Config) -> Path:
    return cfg.root / "data" / cfg.dataset_id


def _experience_cache(cfg: Config) -> Path:
    return cfg.root / "data" / "cache" / cfg.dataset_id / "experiences.jsonl"


def _contact_params(cfg: Config) -> ContactParams:
    return ContactParams(
        px_per_mm=cfg.geometry.px_per_mm,
        closing_axis="x",
        pad_length_mm=cfg.geometry.pad_length_mm,
        minimum_bend_radius_mm=cfg.geometry.minimum_bend_radius_mm,
        side_angle_deg=cfg.geometry.side_angle_deg,
        minimum_contact_fraction=cfg.geometry.minimum_contact_fraction,
    )


def _object_already_exists(dataset_root: Path, object_id: str) -> bool:
    if (dataset_root / "objects" / object_id).exists():
        return True
    dataset_csv = dataset_root / "dataset.csv"
    if not dataset_csv.is_file():
        return False
    with dataset_csv.open(newline="", encoding="utf-8") as fh:
        return any(slug(row.get("Object", "")) == object_id for row in csv.DictReader(fh))


def _append_dataset_row(
    dataset_root: Path,
    *,
    object_name: str,
    object_id: str,
    mass_g: float,
    roughness_index: float,
    contact: float,
    outcomes: dict[Gripper, tuple[bool, float | None]],
) -> None:
    dataset_csv = dataset_root / "dataset.csv"
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not dataset_csv.exists()
    silicone_feasible, silicone_force = outcomes[Gripper.SILICONE]
    gecko_feasible, gecko_force = outcomes[Gripper.GECKO]
    candidates = {
        gripper.value: force
        for gripper, (feasible, force) in outcomes.items()
        if feasible and force is not None
    }
    if not candidates:
        favored = "none"
    else:
        minimum = min(candidates.values())
        winners = [name for name, force in candidates.items() if force == minimum]
        favored = winners[0] if len(winners) == 1 else "tie"
    row = {
        "Object": object_name,
        "Image": f"objects/{object_id}/image.png",
        "Mass_g": mass_g,
        "roughness_index": roughness_index,
        "projected_contact_fraction": round(contact, 4),
        "silicone_force_n": silicone_force if silicone_feasible else "",
        "silicone_feasible": silicone_feasible,
        "gecko_force_n": gecko_force if gecko_feasible else "",
        "gecko_feasible": gecko_feasible,
        "favored_gripper": favored,
    }
    with dataset_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATASET_HEADER, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def collect_mock(cfg: Config, n: int, coarse: float, fine: float) -> None:
    import cv2

    bench, devices = make_mock_bench(cfg)
    dataset_root = _dataset_root(cfg)
    objects_root = dataset_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    experience_path = _experience_cache(cfg)
    objects: list[MockObject] = synthetic_objects(cfg, n)

    for obj in objects:
        if _object_already_exists(dataset_root, obj.object_id):
            raise FileExistsError(f"object {obj.object_id!r} already exists in {cfg.dataset_id}")
        bench.set_object(obj)
        # capture + describe once per object (shared image across grippers)
        rgb = devices.camera.capture_rgb()
        object_dir = objects_root / obj.object_id
        object_dir.mkdir(parents=True)
        image_path = object_dir / "image.png"
        if not cv2.imwrite(str(image_path), rgb):
            raise OSError(f"could not write {image_path}")
        image_rel = str(image_path.relative_to(cfg.root))
        contact = obj.projected_contact_fraction
        description = describe(rgb, cfg).description
        mass_g = devices.mass.read_g()
        roughness = devices.roughness.read_index()

        summary = []
        outcomes: dict[Gripper, tuple[bool, float | None]] = {}
        for gripper in GRIPPERS:
            bench.mounted_gripper = gripper
            feasible, min_force, trials = measure_pair(
                devices, cfg, cfg.collection.repeats, coarse, fine
            )
            record = _build_record(
                cfg, obj.object_id, image_rel, mass_g, roughness, contact, gripper,
                description, feasible, min_force, trials, pad_id="mock",
                contact_source="synthetic",
            )
            append_experience(experience_path, record)
            outcomes[gripper] = (feasible, min_force)
            summary.append(f"{gripper.value}={'infeasible' if not feasible else f'{min_force}N'}")
        _append_dataset_row(
            dataset_root,
            object_name=obj.object_id,
            object_id=obj.object_id,
            mass_g=mass_g,
            roughness_index=roughness,
            contact=contact,
            outcomes=outcomes,
        )
        print(f"{obj.object_id}: mass={mass_g:.0f}g rough={roughness} a={contact:.2f}  " + " ".join(summary))
    print(f"Wrote {len(objects)} objects to {dataset_root}")


def collect_real(cfg: Config, coarse: float, fine: float, port: str | None) -> None:
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
    dataset_root = _dataset_root(cfg)
    objects_root = dataset_root / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    experience_path = _experience_cache(cfg)
    print("Loading contact-model segmentation session...")
    segmentation_session = create_rembg_session()

    while True:
        object_name = input("Object id (blank to finish): ").strip()
        if not object_name:
            break
        object_id = slug(object_name)
        if _object_already_exists(dataset_root, object_id):
            print(f"Object {object_id!r} already exists; choose a new id.")
            continue
        input("Place object at the centered grasp location, then press Enter...")
        rgb = devices.camera.capture_rgb()
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{object_id}-", dir=objects_root
            ) as temporary:
                staged_object = Path(temporary)
                staged_image = staged_object / "image.png"
                if not cv2.imwrite(str(staged_image), rgb):
                    raise OSError(f"could not write {staged_image}")
                estimate, contact_summary, _ = analyze_image(
                    staged_image,
                    staged_object / "contact_fraction",
                    object_id,
                    _contact_params(cfg),
                    session=segmentation_session,
                )
                if not estimate.feasible or not estimate.pair.antipodal:
                    raise RuntimeError(
                        "contact model found no feasible antipodal grasp; reposition the object"
                    )
                description = describe(rgb, cfg).description
                staged_object.replace(objects_root / object_id)
        except Exception as exc:  # operator can reposition and retry safely
            print(f"Contact analysis failed for {object_id}: {exc}")
            continue

        results = contact_summary["results"]
        contact = float(results["combined_contact_fraction"])
        image_path = objects_root / object_id / "image.png"
        image_rel = str(image_path.relative_to(cfg.root))
        summary_rel = str(
            (objects_root / object_id / "contact_fraction" / "summary.json").relative_to(
                cfg.root
            )
        )
        mass_g = devices.mass.read_g()
        roughness = devices.roughness.read_index()

        outcomes: dict[Gripper, tuple[bool, float | None]] = {}
        for gripper in GRIPPERS:
            input(f"Mount the {gripper.value.upper()} pad, clean it, then press Enter...")
            pad_id = input(f"{gripper.value} pad id: ").strip() or None
            feasible, min_force, trials = measure_pair(
                devices, cfg, cfg.collection.repeats, coarse, fine
            )
            record = _build_record(
                cfg, object_id, image_rel, mass_g, roughness, contact, gripper,
                description, feasible, min_force, trials, pad_id,
                contact_source="projected_two_pad_v2",
                contact_schema_version=int(contact_summary["schema_version"]),
                contact_summary_path=summary_rel,
                contact_grasp_feasible=bool(results["grasp_feasible"]),
                contact_antipodal_grasp=bool(results["antipodal_grasp"]),
                contact_floor_applied=bool(results["contact_floor_applied"]),
            )
            append_experience(experience_path, record)
            outcomes[gripper] = (feasible, min_force)
            print(f"  {gripper.value}: {'INFEASIBLE' if not feasible else f'{min_force} N'}")
        _append_dataset_row(
            dataset_root,
            object_name=object_name,
            object_id=object_id,
            mass_g=mass_g,
            roughness_index=roughness,
            contact=contact,
            outcomes=outcomes,
        )
    print(f"Appended objects to {dataset_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ground-truth minimum-force data.")
    p.add_argument("--mock", action="store_true", help="Use the physics-backed mock bench.")
    p.add_argument("--n", type=int, default=30, help="Number of synthetic objects (mock only).")
    p.add_argument("--port", default=None, help="Gripper serial port (real mode).")
    p.add_argument(
        "--dataset", default="collected",
        help="Dataset folder under data/ (default: collected).",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument(
        "--confirm-gemini-cost",
        action="store_true",
        help="Required because every collected object receives a Gemini descriptor.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_gemini_cost:
        raise SystemExit("collection requires --confirm-gemini-cost")
    cfg = load_config(args.config) if args.config else load_config()
    cfg.dataset_id = slug(args.dataset)
    coarse = cfg.collection.coarse_step_n
    fine = cfg.collection.fine_step_n
    if args.mock:
        collect_mock(cfg, args.n, coarse, fine)
    else:
        collect_real(cfg, coarse, fine, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
