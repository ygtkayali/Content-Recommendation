# Task Stream (Priority + Dependency)

Use this as the active execution board.

## Status Keys
- `READY` — can be started now.
- `BLOCKED` — waiting on dependencies.
- `IN-PROGRESS` — currently being worked on.
- `DONE` — finished with linked evidence.

## Priority Keys
- `P0` critical path
- `P1` high value
- `P2` useful but deferrable
- `P3` nice to have

## Active Queue

| ID | Task | Lane | Priority | Depends On | Status | Owner | Evidence |
|---|---|---|---|---|---|---|---|
| T-008 | Define web app MVP scope from ML artifacts | Plan | P0 | T-007A | READY | you | docs/web/plan.md |
| T-009 | Design frontend data contract for recommendations | Plan | P0 | T-008 | READY | you | docs/shared/interface-contract.md |
| T-010 | Plan search UX and recommendation results UI | Plan | P1 | T-008 | READY | you | docs/web/plan.md |
| T-011 | Define API loading strategy for exported artifacts | Plan | P1 | T-009 | READY | you | ml/artifacts/content_based_anime_v1 |
| T-012 | Create web implementation task breakdown | Execution | P1 | T-010 | READY | you | docs/ops/task-stream.md |

## Pull Rules
1. Always pick highest-priority `READY` task.
2. If no task is `READY`, resolve blockers in dependency order.
3. If a task grows too large, split into child tasks and update dependency links.
4. Move `DONE` tasks to the archive section monthly to keep board short.

## Archive (Completed)
| ID | Task | Completed On | Evidence |
|---|---|---|---|
| T-001 | Finalize canonical anime schema v1 | 2026-03-04 | data/processed/anime.parquet |
| T-002 | Save cleaned anime dataset to `data/processed/` | 2026-03-04 | data/processed/anime.parquet |
| T-003A | Build anime content-based baseline notebook | 2026-03-04 | ml/experiments/content_based_filtering.ipynb |
| T-004A | Add manual weighted scoring | 2026-03-04 | ml/experiments/content_based_filtering.ipynb |
| T-005A | Add Bayesian rating (normalized) | 2026-03-04 | ml/experiments/content_based_filtering.ipynb |
| T-006A | Add MMR post-processing | 2026-03-04 | ml/experiments/content_based_filtering.ipynb |
| T-007A | Export artifacts for web usage | 2026-03-04 | ml/artifacts/content_based_anime_v1 |
