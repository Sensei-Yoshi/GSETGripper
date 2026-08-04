# Force-Prediction pre-E2-removal archive

This directory is a read-only, Git-tracked archive of generated Force-Prediction
artifacts that were removed from the active application when experiment E2 and
legacy artifact compatibility were retired.

## Original locations

- `data/MatForceFinal/results`
- `data/MatForceFinal/runs`
- `data/MatForceFinal/run_images`
- `data/MatForceFinal/suites`
- `data/cache/generation`
- `data/cache/MatForceFinal/generation`

The archived directory layout below `data/` preserves these relative paths. The
`provenance/snapshots/` directory records the active configuration, prompts, dataset,
and preparation manifest at archival time. Git revision, status, and the pre-archive
working-tree patch are recorded alongside those snapshots.

Artifact identifiers beginning with `20260804` use UTC timestamps. They were generated
on August 3, 2026 in America/New_York.

## Archived schema versions

- Saved single-run artifacts use schema version 9.
- Result JSON spans benchmark schemas 9 and 10; each file's `schema_version` and
  `artifact_type` fields are authoritative.
- Suite manifests use schema version 11.
- Generation-response cache entries are content-addressed responses and do not carry an
  application artifact schema version.

These versions describe the pre-removal formats and are intentionally incompatible with
the active application schemas. No migration or read-through is performed.

The active application must not discover, load, update, or write files in this archive.
Use `FILE-INVENTORY.txt` and `SHA256SUMS` to verify its contents.
