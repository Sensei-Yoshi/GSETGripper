# Backlog / TODO (post-MVP)

Ordered roughly by expected value. The MVP deliberately uses simple, defensible
choices; each item below is a concrete upgrade path with the file it touches.

## Retrieval  (`retrieval.py`)
- Learn the four hybrid weights (currently hand-set) via cross-validated search.
- Pairwise learning-to-rank trained around *force* similarity.
- Gripper-specific retrieval weights.
- Include the RGB image in the embedding (the client already supports it) rather
  than relying only on the VLM-generated description text.

## Physics  (`physics.py`)
- Gecko seating study: determine whether the minimum-force protocol reaches the
  saturated adhesion regime; if not, `N50` matters and the full nonlinear
  seating term should stay. Consider per-class `N50`.
- Revisit smooth-vs-per-class coefficients once ≥ ~200 objects give dense class
  coverage.

## Objective & safety  (`prediction.py`, `contracts.py`)
- Selection objective beyond `min F*`: damage risk, adhesive contamination, pad
  wear, release difficulty, tool-switch time, prediction uncertainty.
- Fragility: predict a safe-force interval `F_min_lift ≤ F ≤ F_max_damage`
  instead of only the minimum.
- Let the VLM emit `compatibility: "unknown"` and abstain rather than inventing
  certainty on ambiguous surfaces (dry vs slightly oily, clean vs dusty).

## Data & ops
- **Pad-wear tracking** (deferred for the sprint): log pad cycle count; add
  reference-object recalibration every ~20 trials to detect gecko degradation.
- Log temperature/humidity per session (`Meta` fields exist) — both silicone
  friction and van der Waals adhesion are sensitive to them.
- Deliberately populate the **crossover region** (objects where predicted gecko
  and silicone forces are within ~0.5 N) so selection/regret are informative.
- Variant grouping in splits for near-identical objects (multiple apples, bottle
  sizes, filled/empty) — `make_folds` already accepts a `variant_groups` map.

## Reproducibility
- `gemini-embedding-2-preview` is a preview model; pin/cache aggressively and
  note the version in the paper. A local open embedder is the fallback if it is
  deprecated mid-project.
