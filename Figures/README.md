# Figures

## MatForceFinal regression figures

Run from the GSET workspace root:

```bash
env/bin/python GSETGripper/Figures/regression_teseting/generate_matforce_regression_figures.py
```

The script treats each measurement condition as a row and compares linear,
quadratic-ridge, and RBF-SVR regressions using `log10(mass)`, recorded LED roughness,
and (when enabled and available) projected contact fraction. Primary accuracy uses nested
validation grouped by `surface_id`, so sibling conditions cannot cross either the outer or
inner new-surface split. Exact-condition interpolation is reported in separate columns.
Every model reports its sample count so missing contact measurements remain visible. It regenerates:

- `matforce_regression_model_comparison.{png,pdf,svg}`
- `matforce_regression_held_out_predictions.{png,pdf,svg}`
- `matforce_two_feature_response_surface.{png,pdf,svg}`
- `matforce_regression_metrics.csv`
- `matforce_regression_loo_predictions.csv`

The response surface is a descriptive fit to all currently labeled observations. Use the
held-out metrics, not the training fit, when estimating performance on new objects.
Predictions are constrained to the physically valid 0–8 N Gecko-force range.

## MatForceFinal relationship figures

Run from the GSET workspace root:

```bash
env/bin/python GSETGripper/Figures/regression_teseting/generate_matforce_relationship_figures.py
```

The script reads `GSETGripper/Force-Prediction/data/MatForceFinal/dataset.csv`, excludes
rows without a Gecko force label, and regenerates:

- `matforce_mass_vs_gecko_force.{png,pdf,svg}`
- `matforce_roughness_vs_gecko_force.{png,pdf,svg}`

Every plotted object uses the same stable number in both figures. The object key identifies
each point. The mass plot uses a logarithmic horizontal scale because the collected masses
span more than three orders of magnitude. Dashed lines are descriptive least-squares fits,
not calibrated force-prediction models.

## Sandpaper roughness figures

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
