from __future__ import annotations

import ast
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class SearchItem(BaseModel):
    id: int
    title: str
    year: int | None = None
    rating: float | None = None
    poster_url: str | None = None


class ContentDetail(BaseModel):
    id: int
    title: str
    english_name: str | None = None
    score: float | None = None
    genres: list[str] = Field(default_factory=list)
    synopsis: str | None = None
    type: str | None = None
    studios: str | None = None
    source: str | None = None
    popularity: int | None = None
    scored_by: int | None = None
    image_url: str | None = None
    aired: str | None = None
    premiered: str | None = None
    status: str | None = None
    rank: int | None = None


class RecommendationItem(BaseModel):
    id: int
    title: str
    score: float
    poster_url: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    artifacts_loaded: bool
    artifact_dir: str
    metadata_rows: int
    embeddings_shape: list[int]
    detail: str


@dataclass
class ArtifactStore:
    artifact_dir: Path
    metadata: pd.DataFrame | None = None
    source_data: pd.DataFrame | None = None
    embeddings: np.ndarray | None = None
    embedding_similarity: np.ndarray | None = None
    feature_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    load_error: str | None = None


def _parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return [part.strip() for part in text.split(",") if part.strip()]


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _jaccard(a: list[str], b: list[str]) -> float:
    a_set = {item.lower() for item in a if item}
    b_set = {item.lower() for item in b if item}
    if not a_set and not b_set:
        return 0.0
    union = a_set.union(b_set)
    if not union:
        return 0.0
    return len(a_set.intersection(b_set)) / len(union)


def _extract_year(row: pd.Series) -> int | None:
    for col in ["Aired", "aired", "year", "Year", "release_year", "Release Year"]:
        if col in row.index:
            value = row[col]
            if pd.isna(value):
                continue
            text = str(value)
            matches = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
            if matches:
                return int(matches[0])
    return None


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _id_column(df: pd.DataFrame) -> str:
    if "anime_id" in df.columns:
        return "anime_id"
    if "anime_index" in df.columns:
        return "anime_index"
    raise ValueError("No usable id column found (expected 'anime_id' or 'anime_index').")


def _resolve_runtime_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _compute_dynamic_weights(query_row: pd.Series) -> dict[str, float]:
    weights = {
        "embedding": 0.55,
        "genre": 0.20,
        "type": 0.10,
        "studio": 0.05,
        "source": 0.05,
        "rating": 0.05,
    }

    total = sum(weights.values())
    if total <= 0:
        return weights

    return {key: value / total for key, value in weights.items()}


def _apply_mmr(
    candidate_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    lambda_mmr: float,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    selected: list[pd.Series] = []
    selected_indices: list[int] = []
    pool = candidate_df.copy().reset_index(drop=True)

    while len(selected) < min(top_k, len(pool)):
        best_pos = None
        best_score = -np.inf

        for pos in range(len(pool)):
            row = pool.iloc[pos]
            row_pos = int(row["row_pos"])
            relevance = float(row["score"])

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                selected_matrix = embeddings[selected_indices]
                candidate_vec = embeddings[row_pos]
                sims = selected_matrix @ candidate_vec
                diversity_penalty = float(np.max(sims))

            mmr_score = lambda_mmr * relevance - (1.0 - lambda_mmr) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_pos = pos

        chosen = pool.iloc[best_pos].copy()
        chosen["mmr_score"] = best_score
        selected.append(chosen)
        selected_indices.append(int(chosen["row_pos"]))
        pool = pool.drop(index=best_pos).reset_index(drop=True)

    return pd.DataFrame(selected).reset_index(drop=True)


def _load_artifacts(artifact_dir: Path) -> ArtifactStore:
    store = ArtifactStore(artifact_dir=artifact_dir)

    try:
        metadata_csv_path = artifact_dir / "anime_metadata.csv"
        metadata_parquet_path = artifact_dir / "anime_metadata.parquet"
        embeddings_path = artifact_dir / "synopsis_embeddings.npy"
        similarity_path = artifact_dir / "embedding_similarity.npy"
        feature_arrays_path = artifact_dir / "feature_arrays.npz"
        config_path = artifact_dir / "config.json"

        if not embeddings_path.exists():
            raise FileNotFoundError("Required artifact missing: synopsis_embeddings.npy")
        if not metadata_csv_path.exists() and not metadata_parquet_path.exists():
            raise FileNotFoundError(
                "Required artifact missing: anime_metadata.csv (or .parquet)"
            )

        if metadata_csv_path.exists():
            metadata = pd.read_csv(metadata_csv_path)
        else:
            metadata = pd.read_parquet(metadata_parquet_path)
        embeddings = np.load(embeddings_path)

        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Artifact mismatch: metadata rows ({len(metadata)}) != embeddings rows ({embeddings.shape[0]})"
            )

        if "anime_index" not in metadata.columns:
            metadata = metadata.reset_index().rename(columns={"index": "anime_index"})

        source_data_env = os.getenv("ANIME_SOURCE_DATA_PATH", "data/processed/anime.csv")
        source_data_path = _resolve_runtime_path(source_data_env, PROJECT_ROOT)

        # Fallback: try .parquet if the configured path doesn't exist
        if not source_data_path.exists() and source_data_path.suffix == ".csv":
            alt = source_data_path.with_suffix(".parquet")
            if alt.exists():
                source_data_path = alt

        # Enrich lightweight artifact metadata with full anime columns when available.
        source_df: pd.DataFrame | None = None
        if source_data_path.exists():
            if source_data_path.suffix == ".parquet":
                source_df = pd.read_parquet(source_data_path)
            else:
                source_df = pd.read_csv(source_data_path)

            join_key = None
            if "anime_id" in metadata.columns and "anime_id" in source_df.columns:
                join_key = "anime_id"
            elif "Name" in metadata.columns and "Name" in source_df.columns:
                join_key = "Name"

            if join_key:
                if join_key == "anime_id":
                    metadata[join_key] = pd.to_numeric(metadata[join_key], errors="coerce")
                    source_df[join_key] = pd.to_numeric(source_df[join_key], errors="coerce")
                else:
                    metadata[join_key] = metadata[join_key].fillna("").astype(str).str.strip().str.lower()
                    source_df[join_key] = source_df[join_key].fillna("").astype(str).str.strip().str.lower()

                metadata_cols = set(metadata.columns)
                source_cols = [c for c in source_df.columns if c not in metadata_cols or c == join_key]
                source_subset = source_df[source_cols].copy()

                metadata = metadata.merge(
                    source_subset,
                    on=join_key,
                    how="left",
                    suffixes=("", "_src"),
                )

                # Fill missing base fields from merged source columns where applicable.
                for col in [
                    "English name",
                    "Score",
                    "Genres",
                    "Synopsis",
                    "Type",
                    "Studios",
                    "Source",
                    "Popularity",
                    "Scored By",
                    "Image URL",
                    "Aired",
                    "Premiered",
                    "Status",
                    "Rank",
                    "anime_id",
                ]:
                    src_col = f"{col}_src"
                    if src_col in metadata.columns:
                        if col in metadata.columns:
                            metadata[col] = metadata[col].where(metadata[col].notna(), metadata[src_col])
                        else:
                            metadata[col] = metadata[src_col]
                        metadata = metadata.drop(columns=[src_col])

        metadata = metadata.sort_values("anime_index").reset_index(drop=True)

        if similarity_path.exists():
            embedding_similarity = np.load(similarity_path)
        else:
            embedding_similarity = None

        feature_arrays: dict[str, np.ndarray] = {}
        if feature_arrays_path.exists():
            loaded = np.load(feature_arrays_path)
            feature_arrays = {key: loaded[key] for key in loaded.files}

        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)

        # Optional normalization fallback for bayesian scores if only raw exists
        if "bayesian_scores_norm" not in feature_arrays and "bayesian_scores" in feature_arrays:
            raw = feature_arrays["bayesian_scores"].astype(float)
            min_v = float(np.min(raw))
            max_v = float(np.max(raw))
            if max_v > min_v:
                feature_arrays["bayesian_scores_norm"] = (raw - min_v) / (max_v - min_v)
            else:
                feature_arrays["bayesian_scores_norm"] = np.zeros_like(raw)

        store.metadata = metadata
        store.source_data = source_df
        store.embeddings = embeddings
        store.embedding_similarity = embedding_similarity
        store.feature_arrays = feature_arrays
        store.config = config
        store.loaded = True
    except Exception as exc:
        store.loaded = False
        store.load_error = str(exc)

    return store


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _get_artifact_dir() -> Path:
    return _resolve_runtime_path(
        os.getenv("ARTIFACT_DIR", "ml/artifacts/content_based_anime_v1"),
        PROJECT_ROOT,
    )


def _get_allowed_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    return origins or ["http://localhost:3000"]


_load_env_file(PROJECT_ROOT / "backend" / ".env")
STORE = ArtifactStore(artifact_dir=_get_artifact_dir())


@asynccontextmanager
async def lifespan(_: FastAPI):
    global STORE
    _load_env_file(PROJECT_ROOT / "backend" / ".env")
    STORE = _load_artifacts(_get_artifact_dir())
    yield


app = FastAPI(
    title="Content Recommendation API",
    version="1.0.0",
    description="Anime-first recommendation backend for portfolio project.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_loaded_store() -> ArtifactStore:
    if not STORE.loaded or STORE.metadata is None or STORE.embeddings is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts are not loaded. {STORE.load_error or 'No details available.'}",
        )
    return STORE


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    metadata_rows = len(STORE.metadata) if STORE.metadata is not None else 0
    embeddings_shape = list(STORE.embeddings.shape) if STORE.embeddings is not None else []

    if STORE.loaded:
        return HealthResponse(
            status="ok",
            artifacts_loaded=True,
            artifact_dir=str(STORE.artifact_dir),
            metadata_rows=metadata_rows,
            embeddings_shape=embeddings_shape,
            detail="Artifacts loaded and ready.",
        )

    return HealthResponse(
        status="degraded",
        artifacts_loaded=False,
        artifact_dir=str(STORE.artifact_dir),
        metadata_rows=metadata_rows,
        embeddings_shape=embeddings_shape,
        detail=STORE.load_error or "Artifacts not loaded.",
    )


@app.get("/api/v1/search", response_model=list[SearchItem], tags=["content"])
def search(
    q: str = Query(..., min_length=1, description="Simple title search text"),
    limit: int = Query(20, ge=1, le=100),
) -> list[SearchItem]:
    store = _require_loaded_store()
    df = store.metadata.copy()
    try:
        id_col = _id_column(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    title_col = _first_existing(df, ["Name", "name", "Title", "title"])
    if not title_col:
        raise HTTPException(status_code=500, detail="Title column not found in metadata.")

    mask = df[title_col].fillna("").astype(str).str.contains(q, case=False, regex=False)
    hits = df[mask].head(limit)

    score_col = _first_existing(df, ["Score", "score", "rating", "Rating"])
    poster_col = _first_existing(df, ["Image URL", "image_url", "poster_url", "poster", "image"])

    result: list[SearchItem] = []
    for _, row in hits.iterrows():
        rating = float(row[score_col]) if score_col and pd.notna(row[score_col]) else None
        poster_url = str(row[poster_col]) if poster_col and pd.notna(row[poster_col]) else None

        result.append(
            SearchItem(
                id=int(row[id_col]),
                title=str(row[title_col]),
                year=_extract_year(row),
                rating=rating,
                poster_url=poster_url,
            )
        )

    return result


@app.get("/api/v1/content/{id}", response_model=ContentDetail, tags=["content"])
def get_content(
    id: int = PathParam(..., ge=0, description="Content ID / anime_index"),
) -> ContentDetail:
    store = _require_loaded_store()
    df = store.metadata
    try:
        id_col = _id_column(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    match = df[df[id_col] == id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Content with id={id} not found.")

    row = match.iloc[0]
    source_row = None
    if store.source_data is not None and not store.source_data.empty:
        sdf = store.source_data
        if "anime_id" in sdf.columns and "anime_id" in row.index and pd.notna(row.get("anime_id")):
            src_match = sdf[pd.to_numeric(sdf["anime_id"], errors="coerce") == pd.to_numeric(row.get("anime_id"), errors="coerce")]
            if not src_match.empty:
                source_row = src_match.iloc[0]
        elif "anime_id" in sdf.columns:
            src_match = sdf[pd.to_numeric(sdf["anime_id"], errors="coerce") == pd.to_numeric(id, errors="coerce")]
            if not src_match.empty:
                source_row = src_match.iloc[0]

        if source_row is None:
            title_for_match = str(row.get("Name", row.get("title", ""))).strip().lower()
            if title_for_match and "Name" in sdf.columns:
                src_match = sdf[sdf["Name"].fillna("").astype(str).str.strip().str.lower() == title_for_match]
                if not src_match.empty:
                    source_row = src_match.iloc[0]

    def value_from(primary_col: str, source_col: str | None = None):
        source_col = source_col or primary_col
        primary_val = row.get(primary_col) if primary_col in row.index else None
        if primary_val is not None and not pd.isna(primary_val):
            return primary_val
        if source_row is not None and source_col in source_row.index:
            source_val = source_row.get(source_col)
            if source_val is not None and not pd.isna(source_val):
                return source_val
        return None

    title_col = _first_existing(df, ["Name", "name", "Title", "title"])
    english_col = _first_existing(df, ["English name", "english_name", "English Name"])
    synopsis_col = _first_existing(df, ["Synopsis", "Synposis", "description", "Description"])
    genres_col = _first_existing(df, ["Genres", "genres", "genre", "Genre", "genre_list"])
    score_col = _first_existing(df, ["Score", "score", "rating", "Rating"])
    votes_col = _first_existing(df, ["Scored By", "scored_by", "Votes", "votes"])
    type_col = _first_existing(df, ["Type", "type"])
    studios_col = _first_existing(df, ["Studios", "studios", "Studio", "studio"])
    source_col = _first_existing(df, ["Source", "source"])
    popularity_col = _first_existing(df, ["Popularity", "popularity"])
    image_col = _first_existing(df, ["Image URL", "image_url", "poster_url", "poster", "image"])
    aired_col = _first_existing(df, ["Aired", "aired"])
    premiered_col = _first_existing(df, ["Premiered", "premiered"])
    status_col = _first_existing(df, ["Status", "status"])
    rank_col = _first_existing(df, ["Rank", "rank"])

    return ContentDetail(
        id=int(row[id_col]),
        title=str(value_from(title_col)) if title_col and value_from(title_col) is not None else str(id),
        english_name=(str(value_from(english_col, "English name")) if english_col and value_from(english_col, "English name") is not None else (str(value_from("English name")) if value_from("English name") is not None else None)),
        score=(float(value_from(score_col, "Score")) if score_col and value_from(score_col, "Score") is not None else (float(value_from("Score")) if value_from("Score") is not None else None)),
        genres=_parse_list(value_from(genres_col, "Genres")) if genres_col and value_from(genres_col, "Genres") is not None else _parse_list(value_from("Genres")),
        synopsis=(str(value_from(synopsis_col, "Synopsis")) if synopsis_col and value_from(synopsis_col, "Synopsis") is not None else (str(value_from("Synopsis")) if value_from("Synopsis") is not None else None)),
        type=(str(value_from(type_col, "Type")) if type_col and value_from(type_col, "Type") is not None else (str(value_from("Type")) if value_from("Type") is not None else None)),
        studios=(str(value_from(studios_col, "Studios")) if studios_col and value_from(studios_col, "Studios") is not None else (str(value_from("Studios")) if value_from("Studios") is not None else None)),
        source=(str(value_from(source_col, "Source")) if source_col and value_from(source_col, "Source") is not None else (str(value_from("Source")) if value_from("Source") is not None else None)),
        popularity=(int(float(value_from(popularity_col, "Popularity"))) if popularity_col and value_from(popularity_col, "Popularity") is not None else (int(float(value_from("Popularity"))) if value_from("Popularity") is not None else None)),
        scored_by=(int(float(value_from(votes_col, "Scored By"))) if votes_col and value_from(votes_col, "Scored By") is not None else (int(float(value_from("Scored By"))) if value_from("Scored By") is not None else None)),
        image_url=(str(value_from(image_col, "Image URL")) if image_col and value_from(image_col, "Image URL") is not None else (str(value_from("Image URL")) if value_from("Image URL") is not None else None)),
        aired=(str(value_from(aired_col, "Aired")) if aired_col and value_from(aired_col, "Aired") is not None else (str(value_from("Aired")) if value_from("Aired") is not None else None)),
        premiered=(str(value_from(premiered_col, "Premiered")) if premiered_col and value_from(premiered_col, "Premiered") is not None else (str(value_from("Premiered")) if value_from("Premiered") is not None else None)),
        status=(str(value_from(status_col, "Status")) if status_col and value_from(status_col, "Status") is not None else (str(value_from("Status")) if value_from("Status") is not None else None)),
        rank=(int(float(value_from(rank_col, "Rank"))) if rank_col and value_from(rank_col, "Rank") is not None else (int(float(value_from("Rank"))) if value_from("Rank") is not None else None)),
    )


@app.get("/api/v1/recommend/{id}", response_model=list[RecommendationItem], tags=["recommendation"])
def recommend(
    id: int = PathParam(..., ge=0, description="Content ID / anime_index"),
    limit: int = Query(10, ge=1, le=100),
    diversify: bool = Query(True, description="Apply MMR post-processing for diversity"),
    lambda_mmr: float = Query(0.7, ge=0.0, le=1.0),
    top_k_mmr: int = Query(40, ge=1, le=500),
) -> list[RecommendationItem]:
    store = _require_loaded_store()
    df = store.metadata
    embeddings = store.embeddings
    try:
        id_col = _id_column(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    query_match = df[df[id_col] == id]
    if query_match.empty:
        raise HTTPException(status_code=404, detail=f"Content with id={id} not found.")

    q_pos = int(query_match.index[0])
    row = df.iloc[q_pos]
    title_col = _first_existing(df, ["Name", "name", "Title", "title"])
    genres_col = _first_existing(df, ["Genres", "genres", "genre", "Genre", "genre_list"])
    type_col = _first_existing(df, ["Type", "type"])
    studios_col = _first_existing(df, ["Studios", "studios", "Studio", "studio"])
    source_col = _first_existing(df, ["Source", "source"])
    poster_col = _first_existing(df, ["Image URL", "image_url", "poster_url", "poster", "image"])

    query_genres = _parse_list(row[genres_col]) if genres_col else []
    query_type = _norm_text(row[type_col]) if type_col else ""
    query_studio = _norm_text(row[studios_col]) if studios_col else ""
    query_source = _norm_text(row[source_col]) if source_col else ""

    n_total = len(df)
    if n_total <= 1:
        return []

    top_k_mmr = max(top_k_mmr, limit)
    n_candidates = min(top_k_mmr, n_total - 1)

    # Embeddings are L2-normalized, so dot product equals cosine similarity.
    sims = embeddings @ embeddings[q_pos]
    sims[q_pos] = -1.0

    top_indices = np.argpartition(sims, -n_candidates)[-n_candidates:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    bayes_norm = store.feature_arrays.get("bayesian_scores_norm")
    if bayes_norm is None or len(bayes_norm) != len(df):
        bayes_norm = np.zeros(len(df), dtype=float)

    query_series = pd.Series({})
    weights = _compute_dynamic_weights(query_series)

    candidate_df = df.iloc[top_indices]

    if genres_col:
        genre_scores = np.array(
            [_jaccard(query_genres, _parse_list(value)) for value in candidate_df[genres_col].tolist()],
            dtype=float,
        )
    else:
        genre_scores = np.zeros(len(top_indices), dtype=float)

    if type_col and query_type:
        type_values = np.array([_norm_text(value) for value in candidate_df[type_col].tolist()], dtype=object)
        type_scores = (type_values == query_type).astype(float)
    else:
        type_scores = np.zeros(len(top_indices), dtype=float)

    if studios_col and query_studio:
        studio_values = np.array([_norm_text(value) for value in candidate_df[studios_col].tolist()], dtype=object)
        studio_scores = (studio_values == query_studio).astype(float)
    else:
        studio_scores = np.zeros(len(top_indices), dtype=float)

    if source_col and query_source:
        source_values = np.array([_norm_text(value) for value in candidate_df[source_col].tolist()], dtype=object)
        source_scores = (source_values == query_source).astype(float)
    else:
        source_scores = np.zeros(len(top_indices), dtype=float)

    rating_scores = bayes_norm[top_indices].astype(float)
    embed_scores = sims[top_indices].astype(float)

    final_scores = (
        weights["embedding"] * embed_scores
        + weights["genre"] * genre_scores
        + weights["type"] * type_scores
        + weights["studio"] * studio_scores
        + weights["source"] * source_scores
        + weights["rating"] * rating_scores
    )

    candidates: list[dict[str, Any]] = []
    for pos, row_pos in enumerate(top_indices.tolist()):
        candidate = df.iloc[int(row_pos)]
        candidates.append(
            {
                "id": int(candidate[id_col]),
                "row_pos": int(row_pos),
                "title": str(candidate[title_col]) if title_col else str(row_pos),
                "score": float(final_scores[pos]),
                "poster_url": str(candidate[poster_col]) if poster_col and pd.notna(candidate[poster_col]) else None,
            }
        )

    ranked = pd.DataFrame(candidates).sort_values("score", ascending=False).reset_index(drop=True)

    if diversify:
        mmr_pool = ranked.head(top_k_mmr)
        ranked = _apply_mmr(mmr_pool, embeddings=embeddings, top_k=limit, lambda_mmr=lambda_mmr)
    else:
        ranked = ranked.head(limit)

    ranked = ranked.drop(columns=["row_pos"], errors="ignore")

    return [RecommendationItem(**row.to_dict()) for _, row in ranked.iterrows()]


@app.get("/api/v1/genre/{genre}", response_model=list[SearchItem], tags=["browse"])
def by_genre(
    genre: str = PathParam(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: Literal["bayesian", "popularity"] = Query("bayesian"),
) -> list[SearchItem]:
    store = _require_loaded_store()
    df = store.metadata.copy()
    try:
        id_col = _id_column(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    title_col = _first_existing(df, ["Name", "name", "Title", "title"])
    genres_col = _first_existing(df, ["Genres", "genres", "genre", "Genre", "genre_list"])
    score_col = _first_existing(df, ["Score", "score", "rating", "Rating"])
    votes_col = _first_existing(df, ["Scored By", "scored_by", "Votes", "votes"])
    poster_col = _first_existing(df, ["Image URL", "image_url", "poster_url", "poster", "image"])

    if not title_col or not genres_col:
        raise HTTPException(status_code=500, detail="Required columns for genre endpoint are missing.")

    genre_norm = genre.strip().lower()
    mask = df[genres_col].apply(lambda value: genre_norm in {g.lower() for g in _parse_list(value)})
    filtered = df[mask].copy()

    if filtered.empty:
        return []

    if sort_by == "bayesian" and "bayesian_scores_norm" in store.feature_arrays:
        bayes_map = pd.Series(store.feature_arrays["bayesian_scores_norm"])
        filtered["_sort"] = filtered[id_col].map(bayes_map)
    else:
        filtered["_sort"] = pd.to_numeric(filtered[votes_col], errors="coerce") if votes_col else 0.0

    filtered = filtered.sort_values("_sort", ascending=False).head(limit)

    return [
        SearchItem(
            id=int(row[id_col]),
            title=str(row[title_col]),
            year=_extract_year(row),
            rating=float(row[score_col]) if score_col and pd.notna(row[score_col]) else None,
            poster_url=str(row[poster_col]) if poster_col and pd.notna(row[poster_col]) else None,
        )
        for _, row in filtered.iterrows()
    ]


@app.get("/api/v1/trending", response_model=list[SearchItem], tags=["browse"])
def trending(
    limit: int = Query(20, ge=1, le=100),
    sort_by: Literal["bayesian", "popularity"] = Query("popularity"),
) -> list[SearchItem]:
    store = _require_loaded_store()
    df = store.metadata.copy()
    try:
        id_col = _id_column(df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    title_col = _first_existing(df, ["Name", "name", "Title", "title"])
    score_col = _first_existing(df, ["Score", "score", "rating", "Rating"])
    votes_col = _first_existing(df, ["Scored By", "scored_by", "Votes", "votes"])
    poster_col = _first_existing(df, ["Image URL", "image_url", "poster_url", "poster", "image"])

    if not title_col:
        raise HTTPException(status_code=500, detail="Title column missing from metadata.")

    if sort_by == "bayesian" and "bayesian_scores_norm" in store.feature_arrays:
        df["_sort"] = pd.Series(store.feature_arrays["bayesian_scores_norm"])
    else:
        df["_sort"] = pd.to_numeric(df[votes_col], errors="coerce") if votes_col else 0.0

    top = df.sort_values("_sort", ascending=False).head(limit)

    return [
        SearchItem(
            id=int(row[id_col]),
            title=str(row[title_col]),
            year=_extract_year(row),
            rating=float(row[score_col]) if score_col and pd.notna(row[score_col]) else None,
            poster_url=str(row[poster_col]) if poster_col and pd.notna(row[poster_col]) else None,
        )
        for _, row in top.iterrows()
    ]
