"""Live end-to-end Gemini test on the PUBLIC Exp-Force dataset (129 real objects).

Single-embodiment public set (image, mass, min grasp force). Purpose: exercise the
ENTIRE live Gemini pipeline on real data before our own collection exists:
descriptor -> Gemini Embedding 2 (asymmetric retrieval) -> hybrid retrieval ->
structured per-gripper force prediction -> force MAE, sanity-checked vs Exp-Force's
reported ~0.43 N.

    python scripts/expforce_testset.py --limit 20            # offline stub
    python scripts/expforce_testset.py --live --limit 20 --k 5   # real Gemini (needs .env key)

Roughness/contact are unknown here, held constant (class 3, a=1.0), so this tests
the retrieval + VLM force pathway (our E3), not physics. Cost bounded by --limit;
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

from force_prediction.config import load_config  # noqa: E402
from force_prediction.contracts import (  # noqa: E402
    CandidateQuery,
    ExperienceRecord,
    Gripper,
    Meta,
    Query,
    load_experiences,
    save_experiences,
)
from force_prediction.evaluation import _force_stats, make_folds  # noqa: E402
from force_prediction.perception import describe  # noqa: E402
from force_prediction.prediction import vlm_predict_gripper  # noqa: E402
from force_prediction.retrieval import ExperienceIndex  # noqa: E402

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


def build_dataset(cfg, limit: int | None, live: bool) -> Path:
    root = cfg.root / "data" / "expforce"
    csv_path = root / "dataset.csv"
    exp_path = root / "experiences.jsonl"
    _download(f"{BASE}/dataset.csv", csv_path)

    rows = list(csv.DictReader(csv_path.open()))
    if limit:
        rows = rows[:limit]

    records: list[ExperienceRecord] = []
    for i, r in enumerate(rows):
        oid = _slug(r["Object"]) or f"object_{i:03d}"
        img_name = r["Image"]
        rel = f"data/expforce/images/{img_name}"
        have_img = live and _download(f"{BASE}/images/{img_name}", cfg.root / rel)
        img = None
        if have_img:
            import cv2

            img = cv2.imread(str(cfg.root / rel))
        desc = describe(img, cfg).description if img is not None else ""
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
            retrieved = index.retrieve(q, qv, Gripper.SILICONE, k=k, exclude_object_id=oid)
            image = None
            if rec.image_path and (cfg.root / rec.image_path).exists():
                import cv2

                image = cv2.imread(str(cfg.root / rec.image_path))
            cq = CandidateQuery(**q.model_dump(), candidate_gripper=Gripper.SILICONE)
            pred = vlm_predict_gripper(cfg, cq, image, retrieved, None,
                                       include_paired=False, include_measured=True)
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
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if not args.live:
        cfg.models.dry_run = True
    cfg.paths.experiences = "data/expforce/experiences.jsonl"
    cfg.paths.images = "data/expforce/images"
    cfg.paths.splits = "data/expforce/splits.json"

    exp_path = cfg.path("experiences")
    if args.rebuild or not exp_path.exists():
        build_dataset(cfg, args.limit, args.live)
    records = load_experiences(exp_path)
    if args.limit:
        records = records[: args.limit]
    print(f"Running {'LIVE Gemini' if args.live else 'offline (dry-run stub)'} "
          f"on {len(records)} objects, k={args.k}\n")
    run(cfg, records, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
