# ML Workspace

This folder contains ML implementation code for the recommendation system.

## Current scope
- Start with **anime-only** pipeline first.
- Keep implementation minimal and modular.
- Expand to movie/tv after anime baseline is stable.

## Structure
- `src/` — core ML/recommendation code.
- `scripts/` — runnable scripts (ingestion, feature build, evaluation).
- `experiments/` — quick experiment notes/results.
- `artifacts/` — generated model/feature outputs (local, usually not committed unless small metadata).

## Growth guideline
Start simple. Add subfolders only when one folder accumulates unrelated concerns.
