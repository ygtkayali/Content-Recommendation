# API Evolution Guide (Future Multi-Content Support)

This document captures how to evolve the current anime-first API into a multi-content architecture (anime + movie + tv) when the time comes.

## Current State (Now)
- Backend is anime-first and artifact-driven.
- Endpoints are already structured in a stable `v1` namespace.
- Data mapping currently prioritizes anime columns and artifact schema.

## Why Change Later
As movie/tv datasets and models are added, each content type may have:
- Different fields and metadata richness.
- Different artifact formats and scoring features.
- Different recommendation logic and ranking priorities.

A single hardcoded schema will become fragile.

## Target Direction
Introduce a **content registry** layer without breaking existing API paths.

### Core Idea
- Keep endpoint paths stable.
- Route requests to content-specific artifact loaders and mappers internally.
- Standardize response payloads while allowing content-specific optional fields.

## Recommended Future Design

### 1) Content Registry
Create a registry configuration that maps content type to:
- artifact directory
- metadata schema mapping
- scorer configuration
- optional post-processing configuration (MMR, reranker)

Example conceptual shape:
- `anime -> ml/artifacts/content_based_anime_v1`
- `movie -> ml/artifacts/content_based_movie_v1`
- `tv -> ml/artifacts/content_based_tv_v1`

### 2) ID Strategy
- Keep externally stable IDs per content type.
- Use internal row positions only for matrix operations.
- Store explicit `external_id -> row_pos` mapping per artifact set.

### 3) Schema Mapping Layer
Add explicit per-content mappers from raw metadata to API response models:
- title mapping
- description/synopsis mapping
- score/vote fields
- image/poster fields
- optional fields (`cast`, `director`, `studio`, etc.)

### 4) Scoring Adapter Layer
Keep a shared score interface but pluggable implementations:
- anime scorer
- movie scorer
- tv scorer

This avoids condition-heavy scoring logic in a single function.

### 5) Versioning and Backward Compatibility
- Keep `api/v1` payload shape backward compatible.
- Introduce additive fields only.
- If breaking changes are required, move to `api/v2`.

## Migration Plan (When You Revisit)
1. Extract current anime logic into `AnimeAdapter` (loader + mapper + scorer).
2. Define generic adapter interface.
3. Add `MovieAdapter` and `TVAdapter` placeholders.
4. Introduce request parameter / route strategy for content type.
5. Add integration tests across content adapters.

## Suggested Trigger to Start This Refactor
Start this refactor when at least one of these is true:
- You add second content type artifacts (movie or tv).
- Endpoint code accumulates repeated `if content_type` branches.
- Schema mismatches begin creating endpoint-specific bug fixes.

## Non-Goals for Now
- Full multi-tenant model serving.
- Separate microservices per content type.
- Heavy orchestration complexity.

Keep it monolithic and modular until scale demands otherwise.
