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

    def analyze(
        self,
        image: Image.Image,
        *,
        num_inference_steps: int,
        seed: int,
    ) -> IntrinsicPrediction:
        assert image.size == (8, 6)
        assert num_inference_steps == 2
        assert seed == 7
        return IntrinsicPrediction(
            albedo_rgb=np.full((3, 4, 3), 127, dtype=np.uint8),
            roughness=np.array([[0.0, 0.25, 0.5, 1.0]] * 3, dtype=np.float32),
            metallicity=np.array([[1.0, 0.5, 0.25, 0.0]] * 3, dtype=np.float32),
            processed_size=(4, 3),
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

    run = run_marigold(
        FakeAnalyzer(),  # type: ignore[arg-type]
        image,
        tmp_path,
        background_remover=FakeBackgroundRemover(),  # type: ignore[arg-type]
        source_label="Test Object",
        dataset_id="sample",
        object_id="object_1",
        source_path="data/sample/test.png",
        num_inference_steps=2,
        seed=7,
    )

    run_dir = Path(run["run_dir"])
    assert run["roughness"]["mean"] == 0.125
    assert run["source"]["dataset_id"] == "sample"
    assert run["model"]["processed_size"] == [4, 3]
    assert run["background_removal"]["model"] == "test-background-model"
    assert run["background_removal"]["foreground_fraction"] == 0.5
    for artifact in run["artifacts"].values():
        assert (run_dir / artifact).is_file()
    assert np.load(run_dir / "roughness.npy").shape == (3, 4)

    saved = list_saved_runs(tmp_path)
    assert len(saved) == 1
    assert saved[0]["run_id"] == run["run_id"]
    assert saved[0]["run_dir"] == str(run_dir)


def test_list_saved_runs_ignores_invalid_metadata(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "metadata.json").write_text("not json")

    assert list_saved_runs(tmp_path) == []
