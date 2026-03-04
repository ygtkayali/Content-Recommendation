# Web ↔ ML Interface Contract (v0)

This document defines how the Web team and ML team collaborate inside one unified project.

## Collaboration Model
- One repository, two planning tracks.
- Shared API contract and data schemas are the integration boundary.
- Teams can work independently as long as they respect contract versions.

## Ownership
- Web team owns UI, frontend UX flows, request validation at edge, and presentation logic.
- ML team owns retrieval/ranking logic, intent understanding, and recommendation quality.
- Shared ownership: API schema, error model, and deployment interface.

## Integration Rules
1. Contract-first changes: propose schema changes before implementation.
2. Backward compatibility for at least one version when changing payloads.
3. Every integration change must update this file and related team docs.

## Initial Endpoints (Draft)
- `GET /api/recommend/similar?tmdbId=<id>&type=movie|tv|anime`
- `POST /api/recommend/intent`

## Response Shape (Shared Draft)
```json
{
  "requestId": "uuid",
  "source": "catalog|on-demand|hybrid",
  "items": [
    {
      "tmdbId": 0,
      "title": "string",
      "type": "movie|tv|anime",
      "score": 0.0,
      "reason": "string"
    }
  ],
  "meta": {
    "latencyMs": 0,
    "modelVersion": "string",
    "snapshotDate": "YYYY-MM-DD",
    "isOnDemandBackfill": false
  }
}
```

## Freshness Behavior (Draft)
- Primary path: serve from latest validated catalog snapshot.
- Fallback path: if requested seed title is missing, run on-demand backfill and return results with `meta.isOnDemandBackfill=true`.
- Backfilled items should be persisted into a future snapshot to reduce repeated cold starts.

## Error Model (Shared Draft)
- `400` invalid input
- `404` source title not found
- `429` rate limit (TMDB or internal throttling)
- `500` internal processing error
