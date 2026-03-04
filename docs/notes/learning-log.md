# Learning Log (Ongoing)

Use this file to capture structured reflections after each meaningful session.

## Template
### Date
### What I worked on
### Design decisions made
### Tradeoffs considered
### What I learned
### What confused me
### Next session priority

---

## Entry 1 — Project Kickoff
### Date
2026-03-03

### What I worked on
Created the initial documentation baseline and project charter.

### Design decisions made
- Prioritized architecture-first workflow.
- Split project into two recommendation capabilities (similarity and NLP intent).

### Tradeoffs considered
- Chose slower, deliberate progression over fast feature-only development.

### What I learned
A clear charter reduces ambiguity and improves execution confidence.

### What confused me
How far to go with ML sophistication in early phases.

### Next session priority
Define API contracts and data models for Phase 1 vertical slice.

---

## Entry 2 — Team-Track Split
### Date
2026-03-03

### What I worked on
Restructured documentation into shared, web, and ml tracks.

### Design decisions made
- Adopted a two-team simulation model in one unified project.
- Added explicit Web ↔ ML interface contract.

### Tradeoffs considered
- More documentation overhead in exchange for clearer ownership and better system-design practice.

### What I learned
Separating responsibilities early reduces ambiguity and makes integration planning concrete.

### What confused me
How strict contract versioning should be in early prototyping stages.

### Next session priority
Define concrete request/response schemas and versioning policy for both recommendation endpoints.

---

## Entry 3 — Anime Baseline + Web Handoff Prep
### Date
2026-03-04

### What I worked on
Built an anime-first content-based baseline using synopsis embeddings, genre/entity signals, Bayesian rating, manual weights, and MMR. Exported artifacts for future web integration.

### Design decisions made
- Dropped TF-IDF keyword path for now due to quality concerns.
- Used `all-mpnet-base-v2` for synopsis semantics.
- Added Bayesian rating as normalized signal to avoid scale dominance.
- Added MMR on a larger candidate pool (`top_k_mmr`) for diversity.

### Tradeoffs considered
- Stayed notebook-first for speed and iteration, accepting that code extraction/refactor will be needed before production serving.

### What I learned
Relevance-only ranking quickly creates near-duplicate recommendations; MMR gives a practical diversity improvement with low implementation complexity.

### What confused me
How best to package artifact loading and model inference boundaries for a clean web-service integration.

### Next session priority
Plan web MVP integration path and define backend contract for consuming exported ML artifacts.
