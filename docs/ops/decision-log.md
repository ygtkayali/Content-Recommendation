# Decision Log

Capture architectural/product decisions and rationale.

## Index
| ADR ID | Title | Status | Date | Supersedes | Link |
|---|---|---|---|---|---|
| ADR-001 | Static type-specific weighted ranker for v1 | Accepted | 2026-03-03 | - | #adr-001 |

---

## ADR-001
### Context
Need a practical, explainable v1 ranking system with mixed feature availability across movie/tv/anime.

### Decision
Use a static, type-specific weighted hybrid ranker for v1 with:
- embedding/text + metadata signals,
- Jaccard entity overlap,
- Bayesian rating prior.

### Consequences
- Faster to implement and debug than learned re-ranking.
- Enables clearer baseline evaluation before v2 complexity.

### Follow-up
Evaluate dynamic weighting and cross-encoder reranker in v2 after baseline quality is measured.
