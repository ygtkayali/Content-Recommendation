# Operations Workflow (Task Stream)

This folder manages day-to-day execution using a continuous task stream (not time-based planning).

## Core Model
- Tasks are prioritized and dependency-aware.
- Work is pulled in sequence from a single active board.
- Lanes organize work by intent, not by calendar.

## Lanes
- `Plan` — approved upcoming tasks in priority order.
- `Execution` — tasks in progress or ready next.
- `Decision` — architectural/product decisions and rationale.
- `Ideas` — raw ideas and future possibilities.

## Files
- `task-stream.md` — single source of truth for active tasks.
- `backlog.md` — parked tasks and future candidates.
- `decision-log.md` — decision records index.
- `idea-inbox.md` — uncommitted ideas.
- `templates.md` — reusable entry templates.

## Rules
1. New work starts in `idea-inbox.md` or `backlog.md`, not directly in active execution.
2. A task can enter `Execution` only if dependencies are resolved.
3. Every completed task links to evidence (doc/code/result).
4. Major tradeoffs must get an entry in `decision-log.md`.
5. Keep task IDs stable; never recycle IDs.
