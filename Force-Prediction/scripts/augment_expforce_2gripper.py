"""Augment the public Exp-Force dataset into a semi-realistic TWO-gripper dataset.

Exp-Force gives (image, mass, min grasp force) for ONE compliant friction gripper.
We use those values only as seeds for a fully synthetic validation fixture, then
synthesize gecko vs silicone forces + roughness + contact so the full pipeline
(gripper selection, roughness/contact factors, crossover) can be exercised on
realistic-looking data *before* our own collection exists.

Modeling choices (documented so this is defensible, not made-up):
  * SILICONE starts from the Exp-Force force and is quantized to our force grid.
    It remains a synthetic proxy for this gripper embodiment, not ground truth.
  * ROUGHNESS class (1 smooth .. 5 rough) and CONTACT fraction (0..1) are inferred
    per object from material/shape keywords in its name.
  * GECKO = silicone * ratio(roughness) * contact_adjust * small_noise, where
    ratio() is fit to James et al.'s real-object TPU gecko/silicone ratios and the
    Fig 2/4 roughness crossover: gecko cheaper on smooth (ratio<1), crosses ~class 3,
    dearer on rough (ratio>1), and can exceed the safe force limit -> infeasible.
    A low contact fraction penalizes gecko more (adhesion needs real contact area).

Outputs (under data/expforce/):
  dataset_2gripper.csv        human-readable augmented table
  experiences_2gripper.jsonl  our ExperienceRecord format (2 rows/object) for the pipeline

    python scripts/augment_expforce_2gripper.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from force_prediction.config import load_config  # noqa: E402
from force_prediction.contracts import (  # noqa: E402
    ExperienceRecord,
    Gripper,
    Meta,
    save_experiences,
)

# James real-object gecko/silicone ratios anchor the smooth end (~0.52-1.03); the
# Fig 2/4 roughness collapse anchors the rough end. Ratio = gecko / silicone.
GECKO_RATIO_BY_CLASS = {1: 0.55, 2: 0.72, 3: 1.00, 4: 1.40, 5: 1.90}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _rng(name: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


# --- roughness inference (1 smoothest .. 5 roughest) ------------------------ #
# Ordered rules; first hit wins. Keywords are lowercased substrings of the name.
_ROUGHNESS_RULES: list[tuple[int, tuple[str, ...]]] = [
    (5, ("marshmallow", "lint roller", "duct tape", "magic tape", "tape roll",
         "raspberry", "blackberry", "strawberry")),          # porous / fibrous / bumpy
    (4, ("box", "carton", "crackers", "cheez", "oreo", "pocky", "muffin",
         "swiss roll", "seasoning", "paper cup", "salt", "sugar", "seaweed",
         "snack package", "umbrella")),                        # cardboard / paper / fabric
    (1, ("glass", "whiskey", "tumbler", "margarita", "champagne", "wine",
         "ceramic")),                                          # glass / glazed ceramic
    (2, ("can", "soda", "red bull", "coke", "monster", "7up", "sunkist", "tuna",
         "soup can", "bottle", "jar", "shaker", "lotion", "shampoo", "mustard",
         "bleach", "cleaner", "oxiclean", "nutella", "peanut butter", "toothpaste",
         "hair wax", "gatsby", "cerave", "barbasol", "adapter", "charger",
         "power bank", "phone", "remote", "peeler", "drill", "mug", "cup",
         "apple", "pear", "lemon", "lime", "orange", "grapefruit", "mandarin",
         "grape", "egg", "bell pepper")),                      # smooth plastic/metal/waxy
    (3, ("tomato", "banana", "gummies", "gummy", "bean paste", "jam",
         "power adapter", "stand", "container")),              # matte / soft skin / mixed
]


def infer_roughness(name: str) -> int:
    low = name.lower()
    for cls, keys in _ROUGHNESS_RULES:
        if any(k in low for k in keys):
            return cls
    return 3  # moderate default


# --- contact fraction inference (0..1) -------------------------------------- #
def infer_contact(name: str) -> float:
    low = name.lower()
    rng = _rng(name + "_a")
    small_thin = ("straw", "single pringles", "shot cup", "blueberry", "raspberry",
                  "blackberry", "grape ", "grapes", "red grape", "green grape")
    round_small = ("ball", "berry", "marshmallow", "egg", "lime", "lemon",
                   "mandarin", "strawberry", "grape tomato")
    round_big = ("apple", "orange", "pear", "grapefruit", "tomato", "pepper",
                 "banana", "onion")
    tall_flat = ("can", "bottle", "jar", "cup", "mug", "glass", "shaker",
                 "container", "box", "carton", "stand", "remote", "phone",
                 "adapter", "charger", "roll", "tube", "toothpaste", "drill",
                 "power bank", "seasoning", "salt", "sugar")
    if any(k in low for k in small_thin):
        base = 0.40
    elif any(k in low for k in round_small):
        base = 0.58
    elif any(k in low for k in round_big):
        base = 0.66
    elif any(k in low for k in tall_flat):
        base = 0.88
    else:
        base = 0.70
    return float(np.clip(base + 0.05 * rng.standard_normal(), 0.25, 1.0))


def _quantize(force: float, cfg) -> float:
    inc = cfg.force.increment_n
    return round(min(cfg.force.limit_n, max(cfg.force.min_n, math.ceil(force / inc) * inc)), 6)


def gecko_force(silicone_n: float, c: int, a: float, name: str) -> float:
    ratio = GECKO_RATIO_BY_CLASS[c]
    contact_adjust = 1.0 + 0.25 * (1.0 - a)          # low contact hurts gecko more
    noise = math.exp(0.10 * _rng(name + "_g").standard_normal())
    return silicone_n * ratio * contact_adjust * noise


def break_quantized_tie(
    silicone_n: float,
    gecko_n: float | None,
    roughness_class: int,
    cfg,
) -> tuple[float, float | None]:
    """Give every synthetic object a strict winner by one force-grid step."""
    if gecko_n is None or not math.isclose(gecko_n, silicone_n):
        return silicone_n, gecko_n

    step = cfg.force.increment_n
    if roughness_class <= 2:
        if gecko_n - step >= cfg.force.min_n:
            gecko_n = round(gecko_n - step, 6)
        else:
            silicone_n = round(min(cfg.force.limit_n, silicone_n + step), 6)
    elif gecko_n + step <= cfg.force.limit_n:
        gecko_n = round(gecko_n + step, 6)
    else:
        silicone_n = round(max(cfg.force.min_n, silicone_n - step), 6)
    return silicone_n, gecko_n


def main() -> int:
    cfg = load_config()
    src = cfg.root / "data" / "expforce" / "dataset.csv"
    out_csv = cfg.root / "data" / "expforce" / "dataset_2gripper.csv"
    out_jsonl = cfg.root / "data" / "expforce" / "experiences_2gripper.jsonl"
    limit = cfg.force.limit_n

    rows = list(csv.DictReader(src.open()))
    records: list[ExperienceRecord] = []
    table: list[dict] = []
    n_gecko_fav = n_sil_fav = n_gecko_infeasible = n_crossover = n_ties_adjusted = 0

    for r in rows:
        name = r["Object"]
        oid = _slug(name)
        mass = float(r["Mass"])
        sil = _quantize(float(r["Gripping Force"]), cfg)
        c = infer_roughness(name)
        a = round(infer_contact(name), 3)
        raw_gecko = gecko_force(sil, c, a, name)
        gecko_feasible = raw_gecko <= limit
        gecko = _quantize(raw_gecko, cfg) if gecko_feasible else None
        was_tied = gecko is not None and math.isclose(gecko, sil)
        sil, gecko = break_quantized_tie(sil, gecko, c, cfg)
        n_ties_adjusted += int(was_tied)

        if gecko is not None:
            if gecko < sil:
                n_gecko_fav += 1
            elif sil < gecko:
                n_sil_fav += 1
            if abs(gecko - sil) <= 0.5:
                n_crossover += 1
            favored = "gecko" if gecko < sil else "silicone"
        else:
            n_gecko_infeasible += 1
            n_sil_fav += 1
            favored = "silicone"

        if gecko is not None and math.isclose(gecko, sil):
            raise AssertionError(f"strict-winner adjustment failed for {name}")

        table.append({
            "Object": name, "Image": r["Image"], "Mass_g": mass,
            "roughness_class": c, "projected_contact_fraction": a,
            "silicone_force_n": sil, "silicone_feasible": True,
            "gecko_force_n": gecko if gecko is not None else "",
            "gecko_feasible": gecko_feasible, "favored_gripper": favored,
        })

        img_rel = f"data/expforce/images/{r['Image']}"
        for gripper, force, feas in (
            (Gripper.SILICONE, sil, True),
            (Gripper.GECKO, gecko, gecko_feasible),
        ):
            records.append(ExperienceRecord(
                object_id=oid, image_path=img_rel, mass_g=mass,
                roughness_class=c, projected_contact_fraction=a, gripper=gripper,
                min_force_n=force if feas else None, feasible=feas,
                failed_at_limit_n=None if feas else limit,
                semantic_description=name, meta=Meta(pad_id="synthetic-2gripper"),
            ))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    save_experiences(out_jsonl, records)

    print(f"Wrote {out_csv}  ({len(table)} objects)")
    print(f"Wrote {out_jsonl}  ({len(records)} rows, 2 per object)")
    print(f"\ngecko-favored: {n_gecko_fav}   silicone-favored: {n_sil_fav}   "
          f"gecko-infeasible: {n_gecko_infeasible}   crossover(<=0.5N): {n_crossover}")
    print(f"quantized ties adjusted by one force-grid step: {n_ties_adjusted}")
    ro006 = [t for t in table if t["roughness_class"] in (1, 2)]
    print(f"smooth objects (class 1-2): {len(ro006)}  -> mostly gecko-favored")
    print("\nsample:")
    print(f"{'object':32} {'rough':>5} {'a':>5} {'sil':>5} {'gecko':>6}  fav")
    for t in table[:14]:
        g = t["gecko_force_n"] if t["gecko_force_n"] != "" else "INF"
        print(f"{t['Object'][:32]:32} {t['roughness_class']:>5} "
              f"{t['projected_contact_fraction']:>5} {t['silicone_force_n']:>5} "
              f"{str(g):>6}  {t['favored_gripper']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
