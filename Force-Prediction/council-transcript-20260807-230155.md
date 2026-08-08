# LLM Council Transcript: Joint vs. Isolated Gripper Force Calls

Generated: 2026-08-07 23:01:55 America/New_York

## Original question

> We don't want to have the VLM choose what gripper to use actually - should we just have one joint call but not ask for any comparison?

## Framed question

A robotics force-prediction app currently makes exactly one Gemini structured-generation call when Gecko and silicone are both active. The request includes the same query image, both embodiment descriptions, and, for E2–E5, retrieved empirical outcomes for both grippers. The joint schema asks for two force/feasibility predictions plus a VLM recommendation and comparison evidence. Python—not the VLM—authoritatively selects the lowest-force feasible gripper.

The user does not want the VLM choosing the gripper. Should the design keep one joint call but remove all comparison/recommendation instructions and fields, asking only for two candidate estimates, or should it use two separate per-gripper VLM calls? The goals are scientifically defensible absolute force estimates and deterministic downstream selection, while considering cost, latency, and prompt-induced coupling. Temperature is zero and generation caching is enabled.

## Advisor responses

### The Contrarian

Use two separate per-gripper calls. A joint call cannot produce scientifically defensible independent candidate estimates, regardless of whether comparison language is removed. Both outputs still share one context and decoding trajectory; the model can anchor one force against the other, enforce an implicit ranking, or normalize values across embodiments. Temperature zero reduces sampling variance but does not remove this structural coupling, and caching only reproduces the same coupled result.

Each call should receive the identical query image and common task instructions, exactly one gripper's embodiment description, and only that gripper's retrieved empirical outcomes. Use a minimal schema containing predicted force, feasibility, and optionally calibrated uncertainty or reasoning—not recommendation, winner, or comparison fields.

Run both calls concurrently to reduce latency, cache them independently, then let Python deterministically apply the feasibility and lowest-force rule. The added token cost is the price of eliminating a major experimental confound. Keep the joint mode only as a labeled ablation.

### The First Principles Thinker

Use two separate per-gripper calls. A joint prompt cannot produce scientifically independent estimates merely by deleting comparison language: both embodiments, evidence sets, and output slots remain in the same context, so one candidate can still anchor or calibrate the other. Temperature zero improves repeatability; it does not remove contextual coupling.

Each call should use the same versioned prompt template, model, decoding settings, query image, and schema, changing only the active gripper's embodiment description and that gripper's retrieved outcomes. Remove every recommendation, comparison, preferred-gripper, and cross-gripper evidence field.

Then apply a fully specified Python policy: reject infeasible candidates, choose the feasible candidate with the lower predicted force, use a predetermined tie tolerance and tie-breaker, and return a defined failure state if neither is feasible. If separate estimates show scale mismatch, correct that with held-out calibration for each gripper—not by exposing the grippers to one another in the prompt.

### The Expansionist

Use two separate per-gripper calls, executed concurrently. A joint prompt that merely deletes recommendation language is not genuinely independent: both embodiments, evidence sets, and output slots still share one attention context, so anchoring, contrast effects, ordering effects, and cross-contamination remain possible.

Build one canonical prompt template and instantiate it once for Gecko and once for silicone. Each call should receive the identical query image and shared object fields, but only that gripper's embodiment description and retrieved empirical outcomes. Return predicted force, feasibility, uncertainty, and evidence tied only to that gripper.

Persist the exact prompt version, model version, retrieval IDs, raw structured response, cache status, and selection-rule version. This creates a useful experimental surface: run a frozen joint-versus-separated ablation and quantify coupling through per-gripper MAE, bias, feasibility accuracy, and order sensitivity. Separate calls should be the production and evaluation default.

### The Outsider

Use two separate per-gripper calls, dispatched in parallel. A joint prompt cannot honestly produce independent estimates because both embodiments and evidence sets share one attention context. Temperature zero only improves repeatability; it does not remove structural coupling.

The hidden requirement is invariance: Gecko's reported force should not change merely because silicone was added to the request. Each call should receive the same query image and experiment metadata, but only one embodiment and only that gripper's retrieved outcomes. Do not mention the other gripper, comparison, selection, or even candidate framing.

Version the prompt, model, retrieval procedure, image preprocessing, schema, and force bounds. Cache each request independently, run them concurrently, and let deterministic Python apply the predefined selection rule. A joint non-comparative call is acceptable only as an optimization after an equivalence-bounded ablation demonstrates negligible drift.

### The Executor

Use two separate per-gripper calls. Implement one canonical `predict_candidate(gripper, image, evidence)` function and invoke it once for Gecko and once for silicone. Each call should contain the identical query image, only the selected gripper's embodiment description, only empirical outcomes eligible for that gripper, and the same output schema. Do not include recommendation, comparison, ranking, or mention of the other gripper.

Run the calls concurrently so wall-clock latency is approximately the slower call. Cache each result independently using the image, embodiment, evidence, prompt version, model version, and generation settings. Then pass both immutable predictions to deterministic Python selection with an explicit tie-breaker.

This costs roughly twice the request overhead, but provides cleaner experimental units, independently cacheable results, simpler failure recovery, and interpretable per-gripper metrics. Keep the current joint mode only as a separately labeled ablation.

## Peer reviews

### Review 1

Response E was strongest because it combined isolation with calibration, tie tolerance, and failure-state requirements. Response A was too permissive about accepting a joint mode after negligible average drift, which could hide condition-specific reversals. All five missed that isolated calls are context-isolated, not statistically independent: they still share model, data, image, and retrieval biases. Evaluation must use paired object/surface analysis, repeated stability tests, model metadata, and explicit retry semantics.

### Review 2

Response A was strongest because it made invariance falsifiable: one gripper's estimate should not depend on whether the other is enabled. Response D's provenance focus did not fully specify the production contract or failure behavior. All five missed that isolation establishes noninterference, not predictive validity. A frozen held-out evaluation, leakage-safe retrieval, equal evidence budgets, units validation, per-gripper calibration, and error intervals remain necessary.

### Review 3

Response E was strongest, with A close behind. Response B stopped too early at minimal schema and caching, missing frozen retrieval, partial failures, selector edge cases, and possible scale mismatch between grippers. All five missed that every upstream input must be controlled: image bytes, preprocessing, index, evidence ordering, model snapshot, prompt version, units, and schema ordering.

### Review 4

Response D was strongest because it made the design auditable and the coupling hypothesis testable. Response B treated isolation as sufficient and missed cross-gripper scale calibration. All five missed the need to define the exact physical quantity: stationary-finger normal force, steady versus peak behavior, units, contact phase, feasibility criterion, and any safety margin. Cached temperature-zero outputs show repeatability, not uncertainty.

### Review 5

Response D was strongest for auditability. Response A's exact invariance criterion was too narrow because provider nondeterminism can cause small changes while identical values can still be wrong. All five missed operational controls: rate limits, timeouts, independent retries, provider request IDs, model-version drift, and explicit incomplete behavior. The selector must never silently choose the only successful call as though both were evaluated.

## Chairman synthesis

### Where the Council Agrees

Use two isolated per-gripper VLM calls. Removing explicit comparison language from a joint prompt does not make its outputs independent. Gecko and silicone would still share the same context and decoding process, allowing anchoring, evidence leakage, and schema-position effects. Temperature zero and caching improve repeatability and cost, but they do not remove prompt-induced coupling.

The VLM should return per-gripper measurements only. Python must remain the sole decision-maker.

### Where the Council Clashes

The only meaningful disagreement is whether a joint call could later become an acceptable optimization. One position permits it after an equivalence-bounded ablation demonstrates negligible prediction drift. The stricter position rejects it for scientific evaluation because even small or inconsistent coupling creates an avoidable confound.

The verdict is stricter: keep isolated calls as the production and evaluation default. A joint call may exist only as an explicitly labeled experimental ablation.

### Blind Spots the Council Caught

Two calls provide noninterference, not automatic scientific validity. Both predictions still share the same model, image pipeline, retrieval system, training-data biases, and potentially correlated errors.

A defensible evaluation must freeze and record the model and provider version, prompt and schema version, original and preprocessed image, retrieval index and evidence, units and physical force definition, raw response, retry history, and selector version. Retrieved examples must be leakage-safe and gripper-scoped. Each gripper requires held-out calibration and paired evaluation by object and surface.

If one call fails, the app must report an incomplete prediction. It must never silently choose the only successful gripper as though both were evaluated.

### The Recommendation

Implement one canonical per-gripper prediction function and invoke it twice concurrently. Each request mentions only one gripper, contains only that gripper's retrieved outcomes, uses the same force definition and schema, and returns predicted force, feasibility, and a structured infeasibility reason. It contains no recommendation, preferred-gripper, comparison, winner, or competition framing.

Cache, retry, time out, and log each call independently. After both calls complete, Python applies a versioned policy: reject infeasible results, choose the lower calibrated force, use a predefined tie tolerance and deterministic tie-breaker, return `neither_feasible` where appropriate, and return `incomplete` if either result is unavailable.

### The one thing to do first

Create a regression test for noninterference, then refactor around it: for a fixed image and frozen evidence, Gecko's request and result must be identical whether silicone is enabled or disabled, and vice versa.

## Anonymization mapping

- Response A: The Outsider
- Response B: The Contrarian
- Response C: The Executor
- Response D: The Expansionist
- Response E: The First Principles Thinker
