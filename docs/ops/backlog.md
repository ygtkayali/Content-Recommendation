# Backlog (Prioritized Candidates)

Use for approved work that is not yet on the active queue.

| ID | Title | Lane | Priority | Dependency Risk | Promotion Trigger | Notes |
|---|---|---|---|---|---|---|
| B-001 | Define offline evaluation set and rubric | Plan | P1 | Medium | After first v1 ranking outputs | relevance/diversity/explainability |
| B-002 | Add rule-based query-aware gating | Plan | P2 | Medium | After static v1 baseline is stable | v2 candidate |
| B-003 | Add cross-encoder top-k reranker | Plan | P2 | High | After latency and quality baseline measured | v2 scope |
| B-004 | Build on-demand missing-title backfill path | Plan | P2 | High | After batch refresh is stable | freshness path |
| B-005 | Add user-history event schema | Ideas | P3 | High | Start of v3 | personalization foundation |

## Promotion Rules
- Promote backlog item to `task-stream.md` only when:
  1) dependencies are identified,
  2) definition of done is clear,
  3) owner is assigned.
