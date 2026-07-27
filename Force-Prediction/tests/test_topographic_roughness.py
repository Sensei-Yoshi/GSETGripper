from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from modules.models.marigold_topo import (
    NormalPrediction,
    list_saved_topography_runs,
    run_marigold_topography,
    run_topographic_roughness,
)


class FakeNormalsAnalyzer:
    model_id = "test/marigold-normals"
    device = "cpu"
    processing_resolution = 64

    def analyze(
        self,
        image: Image.Image,
        *,
        num_inference_steps: int,
        ensemble_size: int,
        seed: int,
    ) -> NormalPrediction:
        assert image.size == (64, 64)
        assert num_inference_steps == 2
        assert ensemble_size == 3
        assert seed == 11
        yy, xx = np.indices((64, 64))
        nx = 0.3 * np.sin(xx * np.pi / 4) * np.sin(yy * np.pi / 4)
        ny = 0.2 * np.cos(xx * np.pi / 4) * np.sin(yy * np.pi / 4)
        nz = np.ones_like(nx)
        normals = np.stack((nx, ny, nz), axis=-1).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return NormalPrediction(
            normals=normals,
            processed_size=(64, 64),
            uncertainty=np.full((64, 64), 0.01, dtype=np.float32),
        )


def _appearance_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "appearance"
    run_dir.mkdir()
    Image.new("RGB", (64, 64), color=(180, 100, 60)).save(run_dir / "inference_crop.png")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:60, 4:60] = 255
    scoring = np.zeros((64, 64), dtype=np.uint8)
    scoring[12:52, 12:52] = 255
    Image.fromarray(mask).save(run_dir / "inference_mask.png")
    Image.fromarray(scoring).save(run_dir / "scoring_mask.png")
    metadata = {
        "run_id": "appearance-run",
        "source": {"dataset_id": "sample", "object_id": "fruit", "label": "Fruit"},
        "artifacts": {
            "inference_crop": "inference_crop.png",
            "inference_mask": "inference_mask.png",
            "scoring_mask": "scoring_mask.png",
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    return run_dir


def test_topographic_roughness_persists_normal_residual_diagnostics(tmp_path: Path) -> None:
    output_root = tmp_path / "topography"
    run = run_topographic_roughness(
        FakeNormalsAnalyzer(),  # type: ignore[arg-type]
        _appearance_run(tmp_path),
        output_root,
        num_inference_steps=2,
        ensemble_size=3,
        seed=11,
    )

    run_dir = Path(run["run_dir"])
    angles = run["topographic_roughness"]["angle_degrees"]
    assert angles["p75"] > 1
    assert run["topographic_roughness"]["score_0_1"] > 0
    assert np.isclose(run["normal_uncertainty"]["mean"], 0.01)
    assert run["source"]["appearance_run_id"] == "appearance-run"
    for artifact in run["artifacts"].values():
        assert (run_dir / artifact).is_file()
    assert not list(run_dir.glob("*.npy"))

    saved = list_saved_topography_runs(output_root)
    assert len(saved) == 1
    assert saved[0]["run_id"] == run["run_id"]


def test_topographic_roughness_rejects_tiny_scoring_mask(tmp_path: Path) -> None:
    appearance = _appearance_run(tmp_path)
    tiny = np.zeros((64, 64), dtype=np.uint8)
    tiny[30:32, 30:32] = 255
    Image.fromarray(tiny).save(appearance / "scoring_mask.png")

    try:
        run_topographic_roughness(
            FakeNormalsAnalyzer(),  # type: ignore[arg-type]
            appearance,
            tmp_path / "topography",
            num_inference_steps=2,
            ensemble_size=3,
            seed=11,
        )
    except RuntimeError as error:
        assert "too small" in str(error)
    else:
        raise AssertionError("expected a tiny scoring mask to be rejected")


def test_direct_topography_reuses_stable_run_directory(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), color=(180, 100, 60))
    kwargs = {
        "source_label": "Stable fruit",
        "dataset_id": "sample",
        "object_id": "fruit",
        "num_inference_steps": 2,
        "ensemble_size": 3,
        "seed": 11,
        "run_key": "streamlit",
    }

    first = run_marigold_topography(
        FakeNormalsAnalyzer(),  # type: ignore[arg-type]
        image,
        tmp_path,
        **kwargs,
    )
    second = run_marigold_topography(
        FakeNormalsAnalyzer(),  # type: ignore[arg-type]
        image,
        tmp_path,
        **kwargs,
    )

    assert first["run_dir"] == second["run_dir"] == str(tmp_path / "streamlit")
    assert second["source"]["object_id"] == "fruit"
    assert second["scoring"]["strategy"] == "eroded_central_principal_axis_band"
    assert len(list_saved_topography_runs(tmp_path)) == 1
    for artifact in second["artifacts"].values():
        assert (Path(second["run_dir"]) / artifact).is_file()
