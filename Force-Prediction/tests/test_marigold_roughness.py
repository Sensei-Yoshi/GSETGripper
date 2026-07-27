from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from modules.models.background_remover import BackgroundRemoval
from modules.models.marigold import IntrinsicPrediction, list_saved_runs, run_marigold


class FakeAnalyzer:
    model_id = "test/marigold"
    device = "cpu"
    processing_resolution = 64
    seen_size: tuple[int, int] | None = None

    def analyze(
        self,
        image: Image.Image,
        *,
        num_inference_steps: int,
        ensemble_size: int,
        seed: int,
    ) -> IntrinsicPrediction:
        self.seen_size = image.size
        assert num_inference_steps == 2
        assert ensemble_size == 3
        assert seed == 7
        width, height = image.size
        return IntrinsicPrediction(
            albedo_rgb=np.full((height, width, 3), 127, dtype=np.uint8),
            roughness=np.full((height, width), 0.4, dtype=np.float32),
            metallicity=np.full((height, width), 0.2, dtype=np.float32),
            processed_size=(width, height),
            roughness_uncertainty=np.full((height, width), 0.1, dtype=np.float32),
            metallicity_uncertainty=np.full((height, width), 0.05, dtype=np.float32),
        )


class FakeBackgroundRemover:
    model_name = "test-background-model"

    def remove(self, image: Image.Image) -> BackgroundRemoval:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
        alpha[:, :4] = 255
        return BackgroundRemoval(
            cutout_rgba=np.dstack((rgb, alpha)),
            alpha=alpha,
        )


def test_run_marigold_persists_diagnostics_and_history(tmp_path: Path) -> None:
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    analyzer = FakeAnalyzer()

    run = run_marigold(
        analyzer,  # type: ignore[arg-type]
        image,
        tmp_path,
        background_remover=FakeBackgroundRemover(),  # type: ignore[arg-type]
        source_label="Test Object",
        dataset_id="sample",
        object_id="object_1",
        source_path="data/sample/test.png",
        num_inference_steps=2,
        ensemble_size=3,
        seed=7,
    )

    run_dir = Path(run["run_dir"])
    assert analyzer.seen_size == (5, 6)
    assert np.isclose(run["roughness"]["mean"], 0.4)
    assert np.isclose(run["roughness_uncertainty"]["mean"], 0.1)
    assert run["source"]["dataset_id"] == "sample"
    assert run["model"]["processed_size"] == [5, 6]
    assert run["model"]["ensemble_size"] == 3
    assert run["background_removal"]["model"] == "test-background-model"
    assert run["background_removal"]["foreground_fraction"] == 0.5
    assert run["crop"]["bbox_xyxy"] == [0, 0, 5, 6]
    for artifact in run["artifacts"].values():
        assert (run_dir / artifact).is_file()
    assert not list(run_dir.glob("*.npy"))

    saved = list_saved_runs(tmp_path)
    assert len(saved) == 1
    assert saved[0]["run_id"] == run["run_id"]
    assert saved[0]["run_dir"] == str(run_dir)


def test_list_saved_runs_ignores_invalid_metadata(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "metadata.json").write_text("not json")

    assert list_saved_runs(tmp_path) == []
