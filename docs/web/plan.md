# Web Plan (Living)

## W0 — Planning Kickoff (Current)
- Define anime-only MVP UX based on available ML artifacts.
- Align request/response payloads with current recommendation outputs.
- Decide first integration mode:
	- Option A: web calls Python API service that loads artifacts.
	- Option B: precomputed recommendations loaded from backend cache.
- Create frontend task breakdown for search input, result list, score display, and error/empty states.
- Track future multi-content API expansion notes in `docs/shared/api-evolution-guide.md`.

## W1 — Foundation
- Initialize Next.js app structure and shared layout.
- Define UI routes and component boundaries.
- Add API client layer matching shared contract.

## W2 — Similar Recommendation UI
- Build title search/selection flow.
- Integrate similar recommendation endpoint.
- Add loading/error/empty states.

### Current Integration Target
- Consume artifacts from `ml/artifacts/content_based_anime_v1` via backend endpoint.
- Use `config.json` for runtime feature/weight metadata and `anime_metadata.parquet` for title mapping.

## W3 — Intent Recommendation UI
- Build prompt input and result rendering flow.
- Add prompt guidance and validation messaging.
- Integrate intent endpoint.

## W4 — Polish & Deployment Readiness
- Improve UX consistency and performance basics.
- Add observability hooks (request IDs surfaced in debug UI/logs).
- Prepare deployment checklist for frontend.
