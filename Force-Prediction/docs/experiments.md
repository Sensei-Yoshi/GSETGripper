# Experiment Protocol

The active suite intentionally contains E1, E2, E4, E5, and E6. E3 and E3B were
removed; the gap is preserved so saved results and research notes are not silently
renumbered.

Every condition runs through `Pipeline(cfg, experiment_id)` under identical frozen,
object-grouped splits. Both gripper rows for an object always remain in the same fold.
All fold-local resources—retrieval indexes, physics calibration, PCA, and residual
regressors—are fit only on training objects.

## Conditions

| ID | Inputs and estimator | Force-generation calls | Research question |
|---|---|---:|---|
| **E1** | Image-only joint VLM | 1 | Can the VLM estimate both forces and recommend a gripper zero-shot from vision plus embodiment descriptions? |
| **E2** | Image, authoritative measurements, joint VLM | 1 | How much do mass, roughness, and optional projected contact improve the zero-shot result? |
| **E4** | Measurements, one paired-object retrieval, joint VLM | 1 | Does empirical paired-gripper experience improve on measured-input zero-shot prediction? |
| **E5** | Fold-calibrated reduced-order physics | 0 | How well do constrained analytical equations with learned physical coefficients perform? |
| **E6** | E5 physics plus a learned semantic residual | 0 | Does flexible residual learning improve on the identical calibrated physics baseline? |

E1, E2, and E4 return both force/feasibility predictions and an explicit VLM
recommendation. The final command is always selected in Python from the lowest feasible
predicted force. Recommendation accuracy and agreement with that selector are reported
separately.

## E4 paired-object retrieval

E2 and E4 receive the same image, measured-property fields, force constraints, and
joint-output schema. E4's only estimator-level addition is experiential learning: a
retrieved list of paired object outcomes and the retrieval metadata needed to interpret
that list.

E4 embeds the query's contact-region description once, ranks objects once, and retrieves
the configured `retrieval.k` objects. Every neighbor carries its Gecko and silicone force
and feasibility outcomes. One structured VLM request receives this shared list and returns
both predictions plus its recommendation. No physics estimate is constructed or sent.

The hybrid retrieval score is not a modified physics equation and does not predict force.
It only ranks candidate neighbors using semantic cosine similarity and closeness in mass,
roughness, and optional projected contact. The force evidence comes from the measured
Gecko/silicone outcomes attached to the retrieved objects.

Projected contact is an explicit Streamlit/config ablation. When disabled, its retrieval
weight becomes zero, the remaining weights are renormalized, and the field is omitted from
E2/E4 VLM payloads and E4 neighbor payloads. E5 and E6 still require contact inside their
physics equations; E6 also removes the standalone contact feature from its residual input.

## E5 versus E6

Both models learn from the training fold:

- E5 fits seven bounded coefficients in fixed analytical holding-force equations: two for
  silicone and five for Gecko. This is calibrated physics/system identification, not an
  untrained formula.
- E6 first fits that same E5 model. It then learns one residual regressor per gripper with
  target `measured_force - physics_force` using log mass, roughness, optional contact,
  physics force, and PCA-compressed semantic embeddings.

At inference, E6 returns `physics_force + predicted_residual`, continuously clamped to the
hardware range. Physics infeasibility remains authoritative. E6 has no neighbor list and
makes no VLM force-generation call.

## Held fixed and reported metrics

The force convention is stationary-finger load-cell normal force, never doubled. Hardware
commands and model outputs are continuous from `0` to `8 N`; collection search resolution
does not quantize predictions. The default neighbor count is the single value
`retrieval.k: 5` in `config.yaml`.

Every experiment reports:

- force MAE, RMSE, median absolute error, and configured threshold accuracy;
- feasibility precision and recall per gripper;
- deterministic gripper-selection accuracy, infeasible-pick rate, and regret;
- raw VLM recommendation accuracy and selector agreement for E1, E2, and E4.

Run one condition with `python scripts/run_experiment.py --exp e4`, or all active
conditions with `python scripts/run_experiment.py --all --dry-run`.
