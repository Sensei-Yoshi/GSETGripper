"""Prepare the 129-object Exp-Force experience pool and semantic retrieval cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from force_prediction.config import load_config  # noqa: E402
from force_prediction.expforce import prepare_dataset, validation_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Download images and use Gemini descriptors.")
    args = parser.parse_args()
    cfg = load_config().model_copy(deep=True)

    print(json.dumps(validation_summary(cfg), indent=2))

    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done:3}/{total}] {name}")

    manifest = prepare_dataset(
        cfg,
        live=args.live,
        progress=progress,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
