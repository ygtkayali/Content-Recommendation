# Backend (FastAPI)

FastAPI service for content search, metadata retrieval, and recommendations using precomputed ML artifacts.

## Endpoints
- `GET /api/v1/search`
- `GET /api/v1/content/{id}`
- `GET /api/v1/recommend/{id}`
- `GET /api/v1/genre/{genre}`
- `GET /api/v1/trending`
- `GET /api/v1/health`

## Artifact expectation
Default artifact path:
- `ml/artifacts/content_based_anime_v1`

Expected files:
- `anime_metadata.parquet`
- `synopsis_embeddings.npy`
- `embedding_similarity.npy` (optional)
- `feature_arrays.npz` (optional)
- `config.json` (optional)

## Run
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Notes
- Artifacts are loaded once at startup.
- If artifacts are missing, health endpoint reports degraded status and model-dependent endpoints return `503`.
