# Roadmap (Living Document)

## Version Alignment
- V1: Web + content similarity.
- V2: Intent-based recommendations.
- V3 (idea stage): user-history personalization and potential two-tower retrieval.

## Phase 0 — Foundation
- Confirm project goals, scope, and success criteria.
- Define architecture draft and module boundaries.
- Set repository conventions and documentation workflow.
- Establish Web/ML ownership and shared interface contract.

## Phase 1 — Vertical Slice: Similar Content (Parallel)
- **ML Track**: TMDB retrieval + baseline hybrid ranking (static type-specific weights, Bayesian prior, Jaccard entity overlap).
- **Web Track**: title search/selection UI + result rendering.
- **Integration**: connect Web flow to `/api/recommend/similar` shared contract.

## Phase 2 — Vertical Slice: NLP Intent (Parallel)
- **ML Track**: prompt intent schema + parser + ranking by constraints, then cross-encoder re-ranking on top-k.
- **Web Track**: prompt input UX + result display + validation/errors.
- **Integration**: connect Web flow to `/api/recommend/intent` shared contract.

## Phase 2.5 — Adaptive Weighting (Post-v1)
- Evaluate rule-based query-aware gating.
- Introduce dynamic/learned weighting only if it improves quality without unacceptable complexity.

## Phase 3 — Quality & Evaluation
- Define lightweight offline evaluation metrics.
- Add manual evaluation rubric (relevance, diversity, explainability).
- Improve ranking logic with observed failure cases.
- Improve frontend UX from observed recommendation behavior.

## Phase 3.5 — Data Freshness & Backfill
- Add scheduled catalog refresh for new/recent content.
- Add on-demand backfill flow for user-searched titles missing from local catalog.
- Add freshness/version observability in recommendation responses.

## Phase 4 — Deployment & Portfolio Packaging
- Deploy the full stack.
- Add project architecture page and demo walkthrough.
- Finalize portfolio/CV narrative around system design decisions.

## Exit Criteria for Each Phase
- Clear artifact produced (doc, feature, endpoint, demo).
- Decision log updated with key tradeoffs.
- Known risks listed and next action defined.
