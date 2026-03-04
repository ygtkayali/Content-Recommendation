# Hybrid Recommendation Architecture Planning (Working Draft)

## Why Hybrid
A pure embedding approach is fast to start, but hybrid scoring usually gives better control and explainability.

## Candidate Scoring Components
- **Metadata similarity score**
  - Genre overlap, keyword overlap, crew/cast overlap, release proximity.
- **Text semantic similarity score**
  - Embedding similarity over descriptions/overviews.
- **Entity overlap score**
  - Director/actor/studio overlap using set-based matching (Jaccard baseline).
- **Intent-constraint match score**
  - How well items satisfy parsed prompt constraints (genre, mood, era, etc.).
- **Popularity/quality prior (optional)**
  - Stabilizes ranking when query signal is weak.

## Feature Representation Strategy (v1)
1. Use mixed representations instead of one representation for everything.
2. Keep schema flexible by content type (`movie`, `tv`, `anime`) since available fields differ.

Suggested v1 defaults:
- Genres: multi-hot/one-hot + overlap score.
- Keywords: text/embedding-based similarity.
- Description: semantic embedding similarity.
- Entities (actors/director/studio): Jaccard overlap baseline in v1.

Entity embeddings can be added later if overlap features are insufficient.

## Draft Hybrid Formula (v1)
Let final score be:

`S = w_meta * S_meta + w_text * S_text + w_intent * S_intent + w_prior * S_prior`

Where weights are static and manually set in v1, versioned, and tuned during evaluation.

Expanded v1 scoring view:
- `S_meta`: genre/keyword/structured metadata similarity.
- `S_text`: description/title semantic similarity.
- `S_entity`: cast/director/studio overlap (Jaccard baseline).
- `S_bayes`: Bayesian-adjusted rating prior (soft prior, no hard cutoff).
- `S_pop`: popularity prior with capped influence.

Final v1 score:
`S_final = w_meta*S_meta + w_text*S_text + w_entity*S_entity + w_bayes*S_bayes + w_pop*S_pop`

## Architecture Pattern (Practical)
1. **Candidate generation**
   - Fast retrieval from metadata/text index.
2. **Feature computation**
  - Compute component scores per candidate, allowing missing features by content type.
3. **Weighted ranking**
  - Apply static v1 weights and return top-k.
4. **Reason generation**
   - Produce `reason` field from dominant score signals.

## Type-Aware Weight Profiles (v1)
- Use separate static weight profiles for `movie`, `tv`, and `anime`.
- Example direction:
  - Movie: higher weight on keywords/genres/director-actor overlap.
  - TV: lower director weight; rely more on genres, keywords, text, cast overlap.
  - Anime: add studio/tag emphasis where available.

This provides context-aware behavior without dynamic gating in v1.

## Freshness Extension
- Maintain a snapshot-backed index for stable serving.
- Support incremental index/embedding updates for newly ingested items.
- Allow on-demand item enrichment for missing seed titles, then merge artifacts into next snapshot.
- Keep ranking behavior consistent across batch and on-demand paths by reusing the same feature transforms.

## Decision Gates (Before Final Lock-In)
- Gate A: embedding-only baseline quality on pilot dataset.
- Gate B: metadata-only baseline quality on pilot dataset.
- Gate C: hybrid weighted baseline vs A/B.
- Proceed with hybrid only if it improves relevance/consistency enough to justify complexity.

## Research Track (Required)
- Review open recommendation architectures (content-based + hybrid ranking patterns).
- Compare retrieval + re-ranking design choices.
- Document tradeoffs: quality, latency, complexity, maintainability.

## Experiment Backlog
- E1: metadata-only cosine/Jaccard baseline.
- E2: text embedding-only baseline.
- E3: simple weighted hybrid.
- E4: constraint-aware re-ranker for prompt intent.

## Dynamic Weighting Roadmap
- v1: static, manually anticipated type-aware weights.
- v2+: dynamic/query-aware gating and possibly learned weighting after enough evaluation data.

## Gated Weights vs Re-Ranker (Important Distinction)
- **Gated/weighted scoring** combines hand-designed feature scores (genre, keywords, entities, text, priors) using explicit weights.
- It is usually easier to debug and explain because each component contribution is visible.
- Even with dynamic gating, this is still mostly a feature-engineered scoring system.

- **Re-ranker (cross-encoder)** is a learned model applied after initial retrieval/ranking.
- It jointly reads query + candidate and learns higher-order relationships that manual feature weights may miss.
- It can improve relevance quality, but costs more latency and is less interpretable than simple weighted scoring.

Practical takeaway:
- Re-ranker is not just a natural extension of gating weights; it is a different modeling layer.
- Recommended order for this project: establish weighted/gated pipeline first, then add cross-encoder re-ranking on top-k in v2. Re-ranker will add another model/pipeline to be created and deployed which increases the complexity.


## Re-Ranker Roadmap
- v1: no cross-encoder in serving path.
- v2: apply cross-encoder re-ranking on top-k candidates after initial retrieval + weighted ranking are stable.
- Keep re-ranker optional only by deployment configuration, but include it as planned V2 architecture.

## Open Questions
- Should dynamic/query-aware gating be rule-based first or directly learned in v2?
- How much explanation quality is needed for portfolio/demo value?
- What latency ceiling should constrain model complexity?

## Deliverables
- Architecture decision record (ADR) for chosen ranking approach.
- Evaluation table comparing baseline and hybrid candidates.
- Versioned scoring config for reproducibility.

                         ┌─────────────────┐
                         │ Query Item      │
                         └─────────────────┘
                                   │
           ┌─────────────── Multi-Signal Similarity ───────────────┐
           │                                                         │
  Text Embedding      Keyword Similarity      Genre Overlap     People Overlap
           │                                                         │
           └───────────────────────┬─────────────────────────────────┘
                                   │
                          Weighted Aggregation
                                   │
                 + Bayesian Rating Signal
                 + Popularity Dampening
                                   │
                             Final Ranking