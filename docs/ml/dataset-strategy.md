# Dataset Strategy (Working Draft)

## Purpose
Define a practical plan to build a satisfiable dataset for recommendations across movies, TV shows, and anime.

## Current Direction
- Start with publicly available datasets (e.g., Kaggle sources) plus TMDB enrichment.
- Combine datasets into one normalized content catalog.
- Prioritize fields needed for similarity first; expand schema iteratively.

## Storage Strategy (v1 Recommendation)
- For v1, file-based storage is enough: keep curated datasets as Parquet and use CSV only for quick inspection/export.
- Use Parquet as the primary format for training/indexing data because it is faster and more compact than CSV.
- Keep immutable snapshots (e.g., `snapshot_date`) so experiments are reproducible.

### When to Introduce a Database
Introduce a database when one or more of these become true:
- Frequent partial updates and concurrent reads/writes are needed.
- You need low-latency filtering/querying for online serving beyond simple file loads.
- You add user history, interactions, and session events that must be updated continuously.
- You need production-grade access control, retention, and operational observability.

### Practical Path
1. v1: Parquet-first offline/nearline pipeline for content-based recommendations.
2. v2: Keep content catalog in Parquet; optionally add lightweight DB for API metadata/cache.
3. v3: Add interaction/event store for user-history-based models and personalization.

## Minimum Required Fields (v1)
- `content_id` (internal unique id)
- `source_id` (tmdb_id or source-specific id)
- `content_type` (`movie|tv|anime`)
- `title`
- `description`
- `genres` (list)
- `keywords` (list)
- `language` (optional in v1 but recommended)
- `release_year` (optional in v1 but recommended)

These fields are enough to start metadata/text-based similarity.

## Candidate Data Sources
- TMDB API (primary metadata source).
- Kaggle movie/TV/anime datasets (supplementary history and breadth).

## Merge & Normalization Plan
1. Ingest each source into a raw table/file format.
2. Standardize names and field shapes (e.g., genre list format).
3. Create unified schema with source provenance.
4. Deduplicate by source id and fuzzy title-year matching fallback.
5. Validate coverage by content type (movie/tv/anime).

## Dataset Freshness & Maintainability
If source datasets have cutoff dates, maintain freshness with two complementary update modes.

### Mode A — Scheduled Catalog Refresh (Batch)
- Run a scheduled ingestion script (daily/weekly) to pull newly trending/popular/latest content.
- Enrich with TMDB fields required by your canonical schema.
- Recompute or incrementally compute features/embeddings for newly added items.
- Write a new snapshot and promote it only after quality checks pass.

### Mode B — On-Demand Backfill (User-Triggered)
- If user searches a title not present in local catalog:
	1. Fetch metadata from TMDB (or another viable API).
	2. Apply the same normalization and feature pipeline used in batch mode.
	3. Compute embedding/features for that single item (or small batch).
	4. Serve recommendation results immediately.
	5. Queue this item for persistence into the next catalog snapshot.

This keeps UX current even when offline snapshots are behind.

## Embedding/Feature Update Strategy
- Keep a `feature_version` and `embedding_version` in artifacts.
- Support incremental computation for new/missing items instead of full re-embedding each run.
- Rebuild full embeddings only when major preprocessing/model changes occur.
- Track per-item processing status (`ready`, `pending`, `failed`) for reliability.

## Serving Consistency Rules
- Prefer snapshot-backed recommendations for stability.
- Allow on-demand fallback path for missing titles.
- Tag outputs with freshness metadata (e.g., `snapshot_date`, `is_on_demand`) for observability.

## Data Quality Checks (v1)
- Null rate for `description`, `genres`, `keywords`.
- Duplicate rate by title-year-type.
- Token-length sanity checks for descriptions.
- Class balance check across content types.

## Risks and Mitigations
- **Schema mismatch across datasets** → define canonical schema early.
- **Sparse keywords in some sources** → backfill from TMDB where possible.
- **Anime metadata inconsistency** → map aliases and track source confidence.
- **Feature drift between batch and on-demand pipelines** → enforce one shared preprocessing module.
- **Latency spikes for on-demand embedding** → use async caching and persist newly computed artifacts.

## Open Questions
- Should anime use TMDB only or mixed with anime-specific datasets?
- What minimum data completeness threshold is acceptable for training/indexing?
- Do we separate retrieval corpus from evaluation corpus from day one?
- At what scale/latency target does file-based serving stop being sufficient?

## Immediate Next Actions
- Shortlist 2–3 concrete datasets and assess schema compatibility.
- Build a one-page data dictionary for the canonical schema.
- Create a small pilot dataset (500–2,000 items) for early architecture tests.
- Define freshness policy (`refresh cadence`, `on-demand fallback enabled`, `max tolerated staleness`).

## Related Documents
- See `docs/ml/data-layout.md` for concrete folder structure, naming convention, and artifact lifecycle.
