# ML Data Layout & Artifact Convention (v1)

This document defines the recommended project data layout to avoid dataset sprawl and inconsistent artifact management.

## Directory Standard

Project root:
- `data/raw/`
- `data/staging/`
- `data/processed/`
- `data/features/`
- `data/snapshots/`
- `data/eval/`

## What Goes Where

### `data/raw/`
- Original downloaded files from Kaggle/TMDB exports.
- No destructive edits.
- Keep source provenance in filename or sidecar metadata.

### `data/staging/`
- Intermediate joins, mapped columns, temporary dedup outputs.
- Safe to delete/rebuild.

### `data/processed/`
- Canonical unified dataset with your schema.
- Deduplicated and validated.
- Input for feature engineering and indexing.

### `data/features/`
- Embeddings and structured feature matrices.
- Store with `feature_version` / `embedding_version` in filename or metadata.

### `data/snapshots/`
- Serving-ready promoted artifacts.
- Includes catalog snapshot + index/scoring artifacts from same build run.

### `data/eval/`
- Query sets, labeled relevance data, evaluation reports.
- Keep separate so experiment quality is reproducible.

## Naming Convention
Prefer deterministic names:
- `<artifact>_<yyyy-mm-dd>_<version>.parquet`
- Example: `catalog_2026-03-03_schema-v1.parquet`
- Example: `embeddings_2026-03-03_emb-v1.parquet`

For snapshot bundles:
- `snapshot_<yyyy-mm-dd>_<ranker-version>/`
- Include a `manifest.json` with artifact references and versions.

## Minimal Metadata Standard
For each promoted artifact/snapshot, track:
- `created_at`
- `source_datasets`
- `schema_version`
- `feature_version`
- `embedding_version`
- `pipeline_commit` (when code repo exists)
- `record_count`

## Lifecycle (Recommended)
1. Ingest into `raw/`.
2. Normalize/dedup through `staging/`.
3. Validate and publish canonical dataset to `processed/`.
4. Build features/embeddings into `features/`.
5. Promote serving bundle to `snapshots/`.
6. Evaluate with fixed inputs in `eval/`.

## Batch + On-Demand Compatibility
- Batch refresh writes full artifacts through standard lifecycle.
- On-demand backfill computes item-level artifacts, serves result, then queues merge into next promoted snapshot.
- Both paths must use shared preprocessing and versioned transforms.

## v1 Guardrails
- Keep v1 simple: Parquet artifacts + manifest files.
- Do not introduce heavy data infrastructure before clear need.
- Add a database/event store later when personalization and high-frequency updates require it.
