"""End-to-end Gemini test on the PUBLIC Exp-Force dataset (129 real objects).

Single-embodiment public set (image, mass, min grasp force). Purpose: exercise the
ENTIRE Gemini pipeline on real data before our own collection exists:
descriptor -> Gemini Embedding 2 (asymmetric retrieval) -> hybrid object retrieval ->
structured joint force prediction -> silicone force MAE, sanity-checked vs Exp-Force's
reported ~0.43 N.

    python scripts/expforce_testset.py --limit 20 --k 5 --confirm-gemini-cost

Roughness/contact are unknown here, held constant (class 3, a=1.0), so this is a
single-embodiment compatibility check of the E4 interface, not a full E4 evaluation.
Cost is bounded by --limit;
all calls disk-cached.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import (  # noqa: E402
    ExperienceRecord,
    Gripper,
    Meta,
    Query,
    load_experiences,
    save_experiences,
)
from modules.evaluation import _force_stats, make_folds  # noqa: E402
from modules.perception import describe  # noqa: E402
from modules.prediction import vlm_predict_joint  # noqa: E402
from modules.retrieval import ExperienceIndex  # noqa: E402

BASE = "https://raw.githubusercontent.com/expforcesubmission/Exp-Force-Website/main"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  warn: could not fetch {url}: {e}", file=sys.stderr)
        return False


def build_dataset(cfg, limit: int | None) -> Path:
    root = cfg.root / "data" / "expforce"
    csv_path = root / "source_dataset.csv"
    exp_path = cfg.root / "data" / "cache" / "expforce" / "single_gripper_experiences.jsonl"
    _download(f"{BASE}/dataset.csv", csv_path)

    rows = list(csv.DictReader(csv_path.open()))
    if limit:
        rows = rows[:limit]

    records: list[ExperienceRecord] = []
    for i, r in enumerate(rows):
        oid = _slug(r["Object"]) or f"object_{i:03d}"
        img_name = r["Image"]
        suffix = Path(img_name).suffix.lower() or ".png"
        rel = f"data/expforce/objects/{oid}/image{suffix}"
        have_img = _download(f"{BASE}/images/{img_name}", cfg.root / rel)
        if not have_img:
            raise RuntimeError(f"could not download required object image: {img_name}")
        import cv2

        img = cv2.imread(str(cfg.root / rel))
        if img is None:
            raise RuntimeError(f"could not decode required object image: {cfg.root / rel}")
        desc = describe(img, cfg).description
        records.append(
            ExperienceRecord(
                object_id=oid,
                image_path=rel if have_img else "",
                mass_g=float(r["Mass"]),
                roughness_class=3,               # unknown in this set -> held constant
                projected_contact_fraction=1.0,  # unknown -> held constant
                gripper=Gripper.SILICONE,        # single embodiment
                min_force_n=float(r["Gripping Force"]),
                feasible=True,
                semantic_description=desc,
                meta=Meta(pad_id="expforce"),
            )
        )
    save_experiences(exp_path, records)
    print(f"Built {len(records)} records at {exp_path}")
    return exp_path


def run(cfg, records, k: int) -> None:
    folds = make_folds(records, cfg)
    pairs: list[tuple[float, float]] = []
    by_id = {r.object_id: r for r in records}
    n_done = 0
    for fold in folds:
        train = [r for r in records if r.object_id in set(fold["train"])]
        index = ExperienceIndex(cfg).fit(train)
        for oid in fold["test"]:
            rec = by_id[oid]
            q = Query(object_id=oid, image_path=rec.image_path, mass_g=rec.mass_g,
                      roughness_class=rec.roughness_class,
                      projected_contact_fraction=rec.projected_contact_fraction,
                      semantic_description=rec.semantic_description)
            qv = index.embed_query(q)  # asymmetric: query-side template
            retrieved = index.retrieve_objects(q, qv, k=k, exclude_object_id=oid)
            image = None
            if rec.image_path and (cfg.root / rec.image_path).exists():
                import cv2

                image = cv2.imread(str(cfg.root / rec.image_path))
            joint = vlm_predict_joint(
                cfg,
                q,
                image,
                retrieved,
                instruction=cfg.prompts.experiments["e4"],
                include_retrieval=True,
                include_measured=True,
            )
            pred = joint.silicone
            assert rec.min_force_n is not None
            pairs.append((rec.min_force_n, pred.predicted_normal_force_n))
            n_done += 1
            print(f"  [{n_done:3}/{len(records)}] {oid[:32]:32}  "
                  f"true={rec.min_force_n:.2f}  pred={pred.predicted_normal_force_n:.2f}")

    stats = _force_stats(pairs, cfg)
    print(f"\nn={stats['n']}  MAE={stats['mae']:.3f} N  RMSE={stats['rmse']:.3f} N  "
          f"medAE={stats['medae']:.3f} N  within0.5N={stats.get('within_0.5n', 0):.2f}")
    print("Exp-Force reported best MAE ~0.43 N (their retrieval-conditioned Gemini).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="Override config.yaml retrieval.k.")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--confirm-gemini-cost", action="store_true")
    args = ap.parse_args()
    if not args.confirm_gemini_cost:
        ap.error("Exp-Force evaluation requires --confirm-gemini-cost")

    cfg = load_config()
    cfg.paths.experiences = "data/cache/expforce/single_gripper_experiences.jsonl"
    cfg.paths.images = "data/expforce/objects"
    cfg.paths.splits = "data/expforce/splits.json"

    exp_path = cfg.path("experiences")
    if args.rebuild or not exp_path.exists():
        build_dataset(cfg, args.limit)
    records = load_experiences(exp_path)
    if args.limit:
        records = records[: args.limit]
    k = args.k if args.k is not None else cfg.retrieval.k
    cfg.retrieval.k = k
    print(f"Running Gemini on {len(records)} objects, k={k}\n")
    run(cfg, records, k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
