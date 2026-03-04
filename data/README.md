# Data Directory Layout

This directory contains all dataset artifacts used by the recommendation pipeline.

## Structure
- `raw/` — immutable source files as downloaded/exported.
- `staging/` — temporary normalized/intermediate files during ETL.
- `processed/` — canonical cleaned datasets used for modeling/indexing.
- `features/` — derived feature tables and embeddings.
- `snapshots/` — versioned promoted catalog snapshots ready for serving.
- `eval/` — evaluation datasets, query sets, and benchmark outputs.

## Rules
1. Keep `raw/` append-only and immutable.
2. Use Parquet for primary artifacts where possible.
3. Never overwrite promoted snapshots; write a new version.
4. Record artifact metadata (schema version, feature version, created timestamp).
5. Keep evaluation datasets separate from training/retrieval artifacts.

## Example Snapshot Naming
- `catalog_2026-03-03_v1.parquet`
- `features_2026-03-03_emb-v1.parquet`
- `snapshot_2026-03-03_hybrid-v1/`
