"""Live Gemini smoke test: image input + JSON-schema structured output.

This exercises the REAL calls the pipeline makes (perception.describe and
prediction.vlm_predict_gripper) against the model in config.yaml, using a photo.

Run it directly to see output:
    python tests/test_gemini_live.py
    python tests/test_gemini_live.py /path/to/some/image.png

Under `pytest` it SKIPS unless `RUN_LIVE_GEMINI_TESTS=1`, an API key is available,
and the default image exists, so the offline suite stays deterministic.

Key handling: put `GEMINI_API_KEY=...` in Force-Prediction/.env (git-ignored) or
`export GEMINI_API_KEY=...`. The model is whatever `models.vlm` is in config.yaml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from force_prediction.llm import get_client, load_dotenv  # noqa: E402

load_dotenv()  # so both pytest's skipif and direct runs see the .env key

import pytest  # noqa: E402

from force_prediction.config import load_config  # noqa: E402
from force_prediction.contracts import (  # noqa: E402
    CandidateQuery,
    Gripper,
    PerGripperPrediction,
)
from force_prediction.perception import describe  # noqa: E402
from force_prediction.prediction import vlm_predict_gripper  # noqa: E402

DEFAULT_IMAGE = (
    "/Users/premshah/Desktop/Robotics/Other/Research/waymo_subset/"
    "extracted_visuals/frames/1550083469645130_front/overlay.png"
)


def _have_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _live_tests_enabled() -> bool:
    return os.environ.get("RUN_LIVE_GEMINI_TESTS") == "1"


def _load_image(path: str):  # noqa: ANN202
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


@pytest.mark.skipif(
    not _live_tests_enabled() or not _have_key() or not Path(DEFAULT_IMAGE).exists(),
    reason="set RUN_LIVE_GEMINI_TESTS=1 with an API key and image to run live",
)
def test_gemini_structured_image() -> None:
    """One real structured-output call with an image -> valid PerGripperPrediction."""
    cfg = load_config()
    img = _load_image(DEFAULT_IMAGE)
    cq = CandidateQuery(
        object_id="probe", image_path=DEFAULT_IMAGE, mass_g=420.0,
        roughness_class=2, projected_contact_fraction=0.83,
        semantic_description="", candidate_gripper=Gripper.GECKO,
    )
    pred = vlm_predict_gripper(cfg, cq, img, [], None, include_paired=False)
    assert isinstance(pred, PerGripperPrediction)
    assert pred.predicted_normal_force_n >= 0


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not _have_key():
        print("No API key found. Put GEMINI_API_KEY=... in Force-Prediction/.env "
              "(or `export GEMINI_API_KEY=...`).")
        return 1
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return 1

    cfg = load_config()
    img = _load_image(image_path)
    print(f"model     : {cfg.models.vlm}")
    print(f"image     : {image_path}\n")

    # (1) Image description via the real descriptor path (image in, JSON out).
    desc = describe(img, cfg)
    print("── describe() ─────────────────────────────────────────────")
    print(f"  description: {desc.description}")
    print(f"  material   : {desc.visible_surface_material}")
    print(f"  condition  : {desc.visible_surface_condition}\n")

    # (2) Full structured per-gripper force prediction (what the pipeline calls).
    print("── vlm_predict_gripper() [note: this photo is not a graspable object] ──")
    for gripper in (Gripper.GECKO, Gripper.SILICONE):
        cq = CandidateQuery(
            object_id="probe", image_path=image_path, mass_g=420.0,
            roughness_class=2, projected_contact_fraction=0.83,
            semantic_description=desc.description, candidate_gripper=gripper,
        )
        pred = vlm_predict_gripper(cfg, cq, img, [], None, include_paired=False)
        print(f"  {gripper.value:8}: force={pred.predicted_normal_force_n} N  "
              f"feasible={pred.feasible}  compat={pred.compatibility.value}")
        print(f"            reason: {pred.reasoning_trace}")

    # (3) Optional: text embedding smoke. Retrieval never embeds image pixels.
    print("\n── embedding smoke ───────────────────────────────────────")
    try:
        vec = get_client(cfg).embed(text="smooth rigid plastic bottle")
        print(f"  embedding dim: {len(vec)} (model {cfg.retrieval.embedding.model})")
    except Exception as e:  # noqa: BLE001
        print(f"  embedding skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
