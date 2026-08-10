# LLM Council Transcript: E6 Contact-Reasoning Redesign

Generated: 2026-08-09 21:46:39 EDT

## Original question

“Should we not have this Stage 1 and Stage 2 thing at all because that's predicting a delta? What's a better way that the VLM should reason for different contact surface areas?”

## Framed question

Should the GSET Force-Prediction E6 method remove its current two-stage VLM reasoning—Stage 1 infers a hypothetical full-contact baseline from retrieved baseline objects and Stage 2 adds a contact penalty transferred from within-surface controlled variants—and what is a scientifically better way for the VLM to reason about different projected contact fractions?

Context: `projected_contact_fraction` is a dimensionless geometry proxy, not actual area. Current retrieved baseline objects often are not at contact 1, but their absolute contact is hidden and their measured force is treated as a full-contact anchor. One creatine controlled pair has `a=1.0, F=2.327 N` and `a=0.1, F=3.961 N`. Recent beaker counterfactual runs violated monotonicity: `a=.50→3.42 N`, `.14→3.961 N`, `.06→2.50 N`. Each run is an independent Gemini call, and Python only validates a nonnegative force and clamps it. The research goal is an honest held-out VLM ablation, not false precision.

## Advisor responses

### The Contrarian

Remove the two-stage method. Its “full-contact baseline” is not merely noisy; it is undefined from the supplied evidence. Stage 1 relabels measurements taken at unknown or non-unit contact as full-contact anchors, while Stage 2 adds a contact effect already embedded in those measurements. Hiding baseline contact fractions prevents the model from detecting this confounding, not from being influenced by it. The decomposition therefore invites double-counting and cannot support causal interpretation.

Do not replace it with a more elaborate prompt-level scaling formula. One controlled creatine pair identifies one finite difference for one object/interface: `F(0.1)-F(1.0)=1.634 N`. It does not identify a universal curve, linearity, additivity, or transferability to a beaker. The observed nonmonotonic counterfactuals show that independent VLM calls are generating plausible numbers rather than applying a stable response function.

A scientifically honest E6 ablation should make one joint, contact-conditioned prediction from all visible evidence. Give the VLM each reference’s mass, roughness, projected-contact proxy, material/geometry description, and force. Explicitly label the proxy as nonphysical and ask it to select genuinely comparable references, reason qualitatively about whether lower proxy values should raise or lower force, and return a broad interval plus confidence—not a fabricated delta. Controlled variants should be shown as local evidence about direction and approximate sensitivity, with transfer uncertainty stated explicitly.

For counterfactual contact sweeps, never use independent calls. Make one call produce all contact conditions simultaneously under a required ordering constraint, or fit a deterministic monotone curve outside the VLM. If only one controlled pair exists, report that quantitative contact dependence is underidentified.

### The First Principles Thinker

Remove the current two-stage method as a quantitative procedure. Its decomposition is not identified by the data. Stage 1 labels measurements taken at unknown/non-unit contact as “full-contact” forces; Stage 2 then adds a contact effect. Therefore contact is already latent in the baseline and may be counted again. The resulting delta has no stable scientific meaning.

Treat every observation as a joint tuple: mass, roughness, material/geometry descriptors, projected-contact proxy, and force. Never hide an observed contact value while using its force as an anchor. Use controlled variants only as local evidence. The creatine pair supports a change for that surface and setup, not a transferable coefficient for beakers.

For a new surface, have the VLM compare the complete query with complete retrieved examples and estimate a range, with uncertainty driven by feature distance and lack of matched contact evidence. Require it to distinguish interpolation, extrapolation, and unsupported transfer. At `a=0.06`, the creatine pair provides at most directional evidence because this lies beyond its observed range.

For counterfactual values, submit all values in one call and request a jointly ordered vector, or enforce monotonicity afterward with a declared deterministic constraint. Contact should be a visible conditioning variable and controlled-pair evidence should inform direction and uncertainty—not become an invented universal additive penalty.

### The Expansionist

Remove the current two-stage method as a quantitative estimator. Its “full-contact baseline” is unidentifiable because the anchors are not full-contact measurements, and the transferred penalty implies precision unsupported by one controlled pair.

A stronger E6 would use joint, evidence-conditioned reasoning. Show each retrieved example’s actual projected contact fraction; retrieve nearest overall observations and same-surface controlled variants; ask the VLM to estimate the query directly from mass, roughness, geometry/material cues, and contact proxy; and distinguish interpolation from extrapolation. Treat the creatine pair as qualitative local evidence, not a universal coefficient. Require a range and confidence label before a point estimate.

For counterfactual evaluation, submit all contact fractions in a single call and require a coherent ordered series. Report both unconstrained outputs and any monotonic projection separately. This creates a better scientific ablation: E6 tests whether visual-semantic reasoning plus transparent analogical evidence improves held-out prediction. As more controlled sweeps become available, they can support surface-specific response shapes or a hierarchical contact model.

### The Outsider

Remove the current two-stage numerical decomposition because its terms lack operational meanings supported by the data. “Full-contact baseline” sounds observed but is inferred from references whose contact may not equal 1. “Contact penalty” sounds causal but is transferred from one object. A reader could mistake prompt-generated arithmetic for an identified physical model.

Rename `projected_contact_fraction` to something like `projected_geometry_ratio` unless experiments establish that it tracks true engaged contact area. Define how it is calculated and why force is expected to change with it. Even monotonic direction should not be treated as universal without physical justification.

Replace E6 with a single joint-evidence prediction. Show complete metadata, clearly mark controlled pairs, ask for comparable references and confounders, infer qualitative contact direction only where supported, estimate force with uncertainty, distinguish interpolation from extrapolation, and state when sensitivity is unidentifiable.

Most importantly, distinguish whether image-based reasoning helps predict held-out force from whether the model has learned a contact-response law. The current dataset may test the first; with one controlled pair, it cannot credibly test the second.

### The Executor

Remove the current two-stage arithmetic. Replace it quickly with a one-stage, complete-context prediction.

Retrieve neighbors using mass, roughness, material/geometry, and projected-contact fraction. Show every record’s actual contact and force. Present controlled pairs separately. Ask directly for `predicted_force`, bounds, nearest analogues, and evidence strength. Tell the VLM that the creatine pair establishes a local trend only. For several fractions of the same object, send them together and require a jointly consistent sequence.

Run three named ablations: E5 contact omitted; E6-joint contact visible with direct one-stage prediction; and E6-joint-monotonic same-object counterfactuals predicted together under the expected ordering. Record raw outputs before correction. Report MAE, monotonicity-violation rate, interval coverage, and repeat-run variance. If deterministic isotonic correction is added, report it separately.

## Anonymization mapping

- Response A: Executor
- Response B: Contrarian
- Response C: Expansionist
- Response D: Outsider
- Response E: First Principles Thinker

## Peer reviews

### Review 1

Response A is strongest because it creates a falsifiable ablation: E5, unconstrained E6-joint, and monotonic E6-joint separate benefits from contact evidence and imposed ordering. Its blind spot is treating monotonicity as scientifically privileged before establishing that the proxy should yield monotonic force. All responses missed repeated physical trials, proxy uncertainty, and leakage-safe surface-family splits.

### Review 2

Response D is strongest because it addresses construct validity. Response A’s largest blind spot is assuming the expected ordering is established; enforcing it could replace hallucinated arithmetic with an unsupported constraint. All five incompletely separate whether the proxy is monotonic with real contact from whether force is monotonic with real contact.

### Review 3

Response A is strongest for executable, falsifiable ablations. Response B’s deterministic monotone curve option risks recreating false precision. All five missed that monotonicity is a scientific hypothesis and that replicated measurements are needed to separate mechanics from VLM instability.

### Review 4

Response D is strongest because it fixes construct validity before optimizing prediction. Response A’s batching and imposed monotonicity change the task and may measure access to a test sequence rather than better pointwise prediction. All five under-separate retrieval from prediction architecture and treat VLM ranges as meaningful without calibration.

### Review 5

Response A is strongest for explicit ablations and raw-output preservation. Response B still asks the VLM for a direct numerical force; opaque joint reasoning is not automatically calibrated. All responses missed using the VLM as an evidence-structuring component while a regularized regression, Gaussian process, or hierarchical model maps structured variables to force and calibrated uncertainty.

## Chairman synthesis

## Where the Council Agrees

The current two-stage E6 should be retired as a quantitative method. Its “full-contact baseline” is not identifiable because its anchors may not represent full contact, and adding a transferred contact delta can double-count contact effects. The creatine pair establishes one local finite difference, not a universal contact-response law.

Every observation should instead remain a complete tuple: force, mass, roughness, projected-contact proxy, geometry/material descriptors, and image evidence. Predictions should distinguish interpolation from extrapolation and acknowledge weak identification.

## Where the Council Clashes

The main disagreements concern what replaces the method: pointwise VLM prediction versus batching fractions; reporting monotonicity versus enforcing it; and letting the VLM predict force versus using it only to structure evidence for a conventional model. Batching can improve coherence but changes the information available. Deterministic monotonic correction is unjustified until monotonicity is empirically established.

## Blind Spots the Council Caught

`projected_contact_fraction` may not measure engaged contact. Pressure distribution, peel, deformation, and failure mode could make the force response nonmonotonic. Evaluation needs replicated force measurements, proxy uncertainty, leakage-safe held-out-surface splits, and repeated VLM runs. VLM-generated ranges are not calibrated intervals; coverage requires calibration or conformal methods.

## The Recommendation

Replace E6 with a one-stage, pointwise **E6-joint** condition:

- reveal actual proxy values for every retrieved observation;
- separate overall analogues from same-object controlled variants;
- label controlled effects as local evidence only;
- request a direct estimate, evidence-strength rating, and interpolation/extrapolation status;
- preserve every raw response.

Compare E5, legacy E6, E6-joint, and a structured-evidence-plus-conventional-model condition. Evaluate MAE, repeat variance, range coverage, and monotonicity diagnostics. Keep batching and monotonic postprocessing as separately reported ablations.

## The One Thing to Do First

Freeze legacy E6 and implement the complete-metadata, pointwise E6-joint prompt while preserving raw outputs.
