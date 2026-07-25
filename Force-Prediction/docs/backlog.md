# Backlog / TODO (post-MVP)

Ordered roughly by expected value. The MVP deliberately uses simple, defensible
choices; each item below is a concrete upgrade path with the file it touches.

## Projected contact fraction  (`perception.py`)
The MVP uses the height-ratio proxy `a = min(1, h_available / h_pad)`, where
`h_available` comes from the near-object vertical span in the depth frame. This
is a geometric proxy, **not** microscopic contact area (especially for gecko).
The separate Streamlit contact-model testbed now provides a width-free,
side-facing two-pad length fraction, but it is deliberately not wired into the
force experiments until it is validated on physical contact measurements.
Upgrade path:
1. **Segmentation overlap** — segment the object and compute pad-rectangle ∩
   object overlap instead of a full-frame vertical span (removes the "object
   spans the frame" framing assumption).
2. **Curvature / local depth** — use Astra+ depth to estimate local curvature at
   the grasp band; widthwise conformity on curved objects dominates real gecko
   contact. Record a coarse `curvature_tag` (flat/cylindrical/complex) now
   (`Meta.curvature_tag` already exists) so it can explain residuals later.
3. **Expected fin-ray deformation** — model how the compliant finger conforms.
4. **Tactile realized contact** — measure actual contact once tactile sensing exists.

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

## Learning  (`learning.py`)
- Add an MLP residual head once unique object–gripper pairs ≫ 200 (below that it
  overfits high-dim embeddings — hence ridge/GBT/GP first).

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
