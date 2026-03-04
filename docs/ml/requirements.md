# ML Requirements (v0)

## Functional Requirements
- Given a TMDB title ID + type, return similar items with scores and explanation hints.
- Given a natural-language prompt, extract intent/constraints and return ranked recommendations.
- Return stable response schema compatible with shared contract.
- Include fallback behavior when signal quality is low.

## Quality Requirements
- Deterministic behavior for same input under same model/config version.
- Track model/ranking version in response metadata.
- Keep median latency within practical interactive limits for web UX.
- Provide basic explainability field (`reason`) for each recommendation.

## Data/Dependency Requirements
- TMDB API retrieval and normalization pipeline.
- Local feature representation for similarity logic.
- Intent parsing strategy (rule-based baseline, model-assisted later if needed).
- Unified dataset covering `movie`, `tv`, and `anime` with canonical schema.
- Minimum required fields for current anime baseline: title, synopsis/description, genres, type, studios, source, score, scored-by votes.
- Dataset quality checks before model/index build (nulls, duplicates, coverage).

## Architecture Requirements
- Start with baseline approaches (metadata-only and embedding-only) before final architecture lock-in.
- Evaluate and document whether hybrid scoring gives meaningful gains over single-approach baselines.
- Keep scoring pipeline modular so new components/weights can be added without API contract changes.
- Version scoring configuration and expose version metadata in responses.

## Team Deliverables
- Retrieval + ranking module design notes.
- Prompt intent schema and parser behavior spec.
- Evaluation rubric and baseline benchmark report.
- Dataset strategy + canonical data dictionary.
- Architecture decision record comparing baseline and hybrid approaches.

## Current Baseline Status (Anime-First)
- Implemented notebook baseline using `all-mpnet-base-v2` synopsis embeddings.
- Implemented genre Jaccard similarity + exact match signals for type/studios/source.
- Implemented Bayesian weighted rating and normalized it to 0-1 before blending.
- Implemented manual weighted scoring and MMR post-processing for diversity.
- Implemented artifact export for web-app integration.
