"""Run selected, resumable preparation stages for one discovered dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.datasets import (  # noqa: E402
    PreparationStage,
    discover_datasets,
    prepare_dataset_stages,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Folder name directly under data/ (defaults to expforce when present).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[stage.value for stage in PreparationStage if stage is not PreparationStage.INDEX],
        default=[PreparationStage.DESCRIPTIONS.value],
        help=(
            "Stages to run. Indexing is automatic; descriptions are added automatically "
            "when embeddings or experiences need them."
        ),
    )
    parser.add_argument(
        "--confirm-gemini-cost",
        action="store_true",
        help="Required when selected stages use Gemini descriptions or embeddings.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered datasets and exit.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config().model_copy(deep=True)
    catalog = discover_datasets(cfg)
    if args.list:
        for dataset in catalog:
            print(
                f"{dataset.dataset_id}\t{len(dataset.objects)} objects\t{dataset.adapter}"
            )
        return 0
    if not catalog:
        raise SystemExit(f"No dataset folders found under {cfg.root / 'data'}")

    by_id = {dataset.dataset_id: dataset for dataset in catalog}
    dataset_id = args.dataset or ("expforce" if "expforce" in by_id else catalog[0].dataset_id)
    if dataset_id not in by_id:
        available = ", ".join(by_id)
        raise SystemExit(f"Unknown dataset {dataset_id!r}. Available datasets: {available}")
    dataset = by_id[dataset_id]
    semantic_stages = {
        PreparationStage.DESCRIPTIONS.value,
        PreparationStage.EMBEDDINGS.value,
        PreparationStage.EXPERIENCES.value,
    }
    if semantic_stages.intersection(args.stages) and not args.confirm_gemini_cost:
        raise SystemExit(
            "Gemini-backed preparation requires --confirm-gemini-cost"
        )

    def progress(done: int, total: int, label: str) -> None:
        print(f"[{done:3}/{total}] {label}")

    manifest = prepare_dataset_stages(
        cfg,
        dataset,
        args.stages,
        progress=progress,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
