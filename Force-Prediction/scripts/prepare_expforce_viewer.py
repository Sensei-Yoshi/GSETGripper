"""Compatibility wrapper for full Exp-Force preparation.

Prefer ``scripts/prepare_dataset.py`` when selecting a dataset or individual stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.datasets import (  # noqa: E402
    PreparationStage,
    get_dataset,
    prepare_dataset_stages,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Download images and use Gemini descriptors.")
    args = parser.parse_args()
    cfg = load_config().model_copy(deep=True)

    dataset = get_dataset(cfg, "expforce")
    print(json.dumps(dataset.summary(), indent=2))

    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done:3}/{total}] {name}")

    manifest = prepare_dataset_stages(
        cfg,
        dataset,
        [
            PreparationStage.DESCRIPTIONS,
            PreparationStage.EMBEDDINGS,
            PreparationStage.EXPERIENCES,
        ],
        live=args.live,
        progress=progress,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
