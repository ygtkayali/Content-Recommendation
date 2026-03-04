# Project Documentation Hub

This folder is the source of truth for planning, architecture, implementation notes, and progress logs for the Content Recommendation System project.

## Why this exists
- Keep project thinking explicit and structured.
- Build a repeatable end-to-end workflow you can reuse in future projects.
- Separate planning/design from code implementation for clarity.

## Suggested reading order
1. `00-project-charter.md` — vision, goals, and high-level methodology.
2. `shared/interface-contract.md` — integration boundary between Web and ML tracks.
3. `shared/version-roadmap.md` — product evolution across V1/V2/V3.
4. `shared/api-evolution-guide.md` — future multi-content API evolution strategy.
5. `01-roadmap.md` — phased execution plan.
5. `ops/task-stream.md` — active priority + dependency execution board.
6. `ops/backlog.md`, `ops/decision-log.md`, `ops/idea-inbox.md` — task stream support lanes.
7. `web/requirements.md` and `web/plan.md` — Web team track.
8. `ml/requirements.md` and `ml/plan.md` — ML team track.
9. `diagrams/architecture-overview.md` — system view and boundaries.
10. `notes/learning-log.md` — ongoing learning and reflections.

## Folder structure
`shared/` — cross-team contracts, roadmap, and future API evolution guides.
- `ops/` — priority/dependency task stream, backlog, decision log, and idea inbox.
- `web/` — web-team requirements and execution plan.
- `ml/` — ml-team requirements and execution plan.
- `diagrams/` — architecture diagrams and sequence flows.
- `notes/` — learning log, decisions, and retrospectives.
- Root docs files — stable planning documents.

## Maintenance rules
- Prefer small, frequent updates over large rewrites.
- Capture major decisions in notes as they happen.
- Keep each file focused on one concern.
