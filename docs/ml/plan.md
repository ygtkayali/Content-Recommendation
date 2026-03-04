# ML Plan (Living)

## Current Execution Snapshot
- Focus narrowed to anime-first implementation before expanding to movie/tv.
- Implemented content-based notebook baseline in `ml/experiments/content_based_filtering.ipynb`.
- Implemented synopsis embedding retrieval, Bayesian scoring, manual feature weights, and MMR re-ranking.
- Implemented artifact export for downstream web integration.

## M0 — Dataset Foundation
- Shortlist and assess candidate movie/tv/anime datasets (including TMDB enrichment strategy).
- Define canonical schema for unified catalog.
- Build pilot merged dataset and run initial quality checks.

## M0.5 — Freshness Pipeline Foundation
- Define dataset refresh cadence and snapshot promotion rules.
- Implement scheduled batch refresh script for new/popular content.
- Implement on-demand backfill flow for missing user-searched titles.
- Ensure batch and on-demand paths share the same preprocessing logic.

## M1 — Baseline Retrieval/Ranking
- Define metadata features from TMDB (genres, keywords, cast/director, overview text).
- Implement metadata-only baseline similarity scoring and candidate retrieval.
- Implement embedding-only baseline over descriptions/overviews.
- Add entity overlap features (actors/director/studio where available) using Jaccard baseline.
- Add Bayesian rating adjustment as soft ranking prior (no hard cutoff).
- Define static type-specific weight profiles for movie/tv/anime.
- Expose API-ready baseline function with contract-compliant output.

### M1 Progress
- Anime synopsis embeddings (`all-mpnet-base-v2`) implemented.
- Genre Jaccard + exact match (type/studios/source) implemented.
- Bayesian rating prior implemented and normalized to 0-1.
- Manual weighted blend implemented.
- MMR post-processing implemented with configurable `top_k_mmr` candidate pool.
- Artifact saving implemented for web consumption.

## M2 — Hybrid Architecture Decision
- Compare metadata-only vs embedding-only vs simple weighted hybrid.
- Document quality/latency/complexity tradeoffs.
- Lock initial hybrid strategy only after evidence from pilot evaluations.
- Keep dynamic/query-aware weighting out of v1 and revisit in v2 after evaluation.

## M3 — Intent Parsing Baseline
- Define intent + constraints schema (genre, mood, era, language, runtime, etc.).
- Implement parser pipeline for user prompts.
- Map parsed constraints to retrieval + re-ranking.

## M4 — Quality Iteration
- Create offline test set with representative queries.
- Measure relevance/diversity manually and with simple metrics.
- Improve ranking weights and parser heuristics using failure analysis.

## M5 — Ops & Maintainability
- Add model/config versioning strategy.
- Add logging hooks for request tracing and debugging.
- Publish runbook for retraining/reconfiguration (if applicable).
- Add freshness metrics (snapshot age, on-demand hit rate, processing failures).

## Immediate Next ML Steps
- Extract notebook logic into reusable script/module for serving.
- Define artifact loading contract with Web team (metadata, embeddings, config).
- Add quick offline checks (relevance spot-check + diversity sanity check).

## Parallel Research Lane
- Study existing recommendation system architectures and document patterns you can adapt.
- Keep a short “adopt / adapt / avoid” list from each research round.

## Future Scope (Post-V2)
- User-history signal integration and similar-user modeling.
- Evaluate two-tower retrieval only after event data and personalization requirements are mature.
