# Sandpaper roughness figures

Run from the GSET workspace root:

```bash
env/bin/python GSETGripper/Figures/generate_sandpaper_roughness_figures.py
```

The script reads the saved Marigold runs under
`GSETGripper/Force-Prediction/test_data/marigold_tests/`, validates the required artifacts,
and regenerates:

- `sandpaper_grit_vs_appearance_roughness.{png,pdf,svg}`
- `sandpaper_appearance_roughness_pipeline.{png,pdf}`
- `sandpaper_roughness_data.csv`

Grit 600 is intentionally excluded from both figures and the exported source table.

Suggested caption for the quantitative figure:

> Mean Marigold IID appearance-roughness estimate versus sandpaper grit. Each point is the
> spatial mean over the segmented central scoring region of one image. Higher grit denotes a
> finer abrasive surface. Grit 600 is omitted. Lines connect observations as a
> visual guide; they do not represent a fitted calibration model.

Suggested caption for the pipeline figure:

> Appearance-roughness estimation for sandpaper samples. ISNet foreground segmentation
> defines the object crop and scoring mask. Marigold IID processes the cropped RGB image and
> predicts an appearance-roughness map, where brighter pixels indicate greater predicted
> roughness. Values report the mean within the scoring region.
