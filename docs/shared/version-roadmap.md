# Version Roadmap (High-Level)

This roadmap keeps product evolution explicit without overcommitting early.

## V1 — Web + Content Similarity (Current Priority)
### Objective
Ship an end-to-end demo where users get similar content from a selected movie/TV/anime title.

### Scope
- Unified content catalog (movie/tv/anime).
- Baseline hybrid similarity retrieval/ranking with static weights.
- Type-specific scoring profiles (movie/tv/anime) to handle schema differences.
- Bayesian rating adjustment and entity-overlap (Jaccard) priors in ranking.
- Web flow for title selection and recommendation display.
- Deployable vertical slice.

### Non-Goals
- Personalized ranking from user behavior.
- Collaborative filtering and similar-user modeling.
- Full real-time on-demand enrichment for out-of-catalog items.
- Learned or dynamic query-aware gating.

## V2 — Intent-Based Recommendations
### Objective
Add prompt-driven recommendation flow that interprets user intent and constraints.

### Scope
- Intent parser + constraint extraction.
- Constraint-aware retrieval/re-ranking.
- Cross-encoder re-ranker for top-k candidates after base pipeline is established.
- Web prompt UX and explainable results.
- Data freshness automation (scheduled catalog refresh + artifact versioning).
- Introduce dynamic/rule-based gating (or learned weighting if data is sufficient).

### Dependency on V1
- Reuse V1 content catalog and serving contracts.

## V3 — User History and Personalization (Idea Stage)
### Objective
Introduce user-behavior signals and personalized ranking.

### Candidate Directions
- User history features (recent views/clicks/likes/watch behavior).
- Similar-user or collaborative signals.
- Two-tower style retrieval architecture (user tower + item tower) if justified by data scale.
- On-demand missing-content enrichment at request time with cache/persist pipeline.

### Preconditions
- Stable event collection pipeline.
- Privacy/data-governance policy.
- Clear online evaluation criteria.

## Decision Policy
- Move to next version only when current version has stable quality, observability, and maintainable operations.
- Keep each version demonstrable and portfolio-ready on its own.
