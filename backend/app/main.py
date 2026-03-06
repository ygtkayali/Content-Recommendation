from __future__ import annotations

import ast
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project root & recommendation pipeline import
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "scripts"))

from recommendation_v1 import (  # noqa: E402
    BASE_FEATURE_WEIGHTS,
    build_runtime_data,
    get_runtime_weights,
    load_artifacts,
    prepare_work_df,
    recommend_by_index,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class SearchItem(BaseModel):
    id: int
    title: str
    year: int | None = None
    rating: float | None = None
    poster_url: str | None = None


class ContentDetail(BaseModel):
    id: int
    title: str
    overview: str | None = None
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    director: str | None = None
    collection: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    runtime: int | None = None
    release_date: str | None = None
    language: str | None = None
    poster_url: str | None = None


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


# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    work_df: pd.DataFrame | None = None
    embeddings: np.ndarray | None = None
    runtime: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    base_weights: dict[str, float] = field(
        default_factory=lambda: BASE_FEATURE_WEIGHTS.copy()
    )
    loaded: bool = False
    load_error: str | None = None
    artifact_dir: str = ""

    # Display-oriented parsed columns (preserving original case)
    display_genres: list[list[str]] = field(default_factory=list)
    display_keywords: list[list[str]] = field(default_factory=list)
    display_actors: list[list[str]] = field(default_factory=list)
    display_directors: list[str] = field(default_factory=list)

    @property
    def metadata_rows(self) -> int:
        return len(self.work_df) if self.work_df is not None else 0

    @property
    def embeddings_shape(self) -> list[int]:
        return list(self.embeddings.shape) if self.embeddings is not None else []


STATE = AppState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_list_display(value: Any) -> list[str]:
    """Parse a stringified list field, preserving original case for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
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


def _poster_url(path: Any) -> str | None:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    s = str(path).strip()
    if not s or s == "nan":
        return None
    if s.startswith("http"):
        return s
    return f"{TMDB_IMAGE_BASE}{s}"


def _extract_year(release_date: Any) -> int | None:
    if release_date is None or pd.isna(release_date):
        return None
    try:
        return int(pd.Timestamp(release_date).year)
    except Exception:
        return None


def _resolve_runtime_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


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
        os.getenv("ARTIFACT_DIR", "ml/artifacts/content_recommendation_v2"),
        PROJECT_ROOT,
    )


def _get_allowed_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    return origins or ["http://localhost:3000"]


def _load_app_state() -> AppState:
    """Load artifacts once and prepare all runtime data."""
    state = AppState()
    artifact_dir = _get_artifact_dir()
    state.artifact_dir = str(artifact_dir)

    try:
        metadata, embeddings, bayesian_scores_norm, recency_scores, config = (
            load_artifacts(artifact_dir)
        )
        work_df = prepare_work_df(metadata)
        runtime = build_runtime_data(work_df, bayesian_scores_norm, recency_scores)
        base_weights = get_runtime_weights(config)

        # Pre-parse display lists (original case) — done once at startup
        display_genres = [_parse_list_display(v) for v in work_df["genres"]]
        display_keywords = [_parse_list_display(v) for v in work_df["keywords"]]
        display_actors = [_parse_list_display(v) for v in work_df["top_5_actors"]]
        display_directors = [
            _parse_list_display(v)[0] if _parse_list_display(v) else ""
            for v in work_df["director"]
        ]

        state.work_df = work_df
        state.embeddings = embeddings
        state.runtime = runtime
        state.config = config
        state.base_weights = base_weights
        state.display_genres = display_genres
        state.display_keywords = display_keywords
        state.display_actors = display_actors
        state.display_directors = display_directors
        state.loaded = True
    except Exception as exc:
        state.loaded = False
        state.load_error = str(exc)

    return state


# ---------------------------------------------------------------------------
# FastAPI bootstrap
# ---------------------------------------------------------------------------
_load_env_file(PROJECT_ROOT / "backend" / ".env")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global STATE
    _load_env_file(PROJECT_ROOT / "backend" / ".env")
    STATE = _load_app_state()
    yield


app = FastAPI(
    title="Content Recommendation API",
    version="2.0.0",
    description="Movie recommendation backend powered by content-based filtering.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_loaded() -> AppState:
    if (
        not STATE.loaded
        or STATE.work_df is None
        or STATE.embeddings is None
        or STATE.runtime is None
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded. "
                f"{STATE.load_error or 'No details available.'}"
            ),
        )
    return STATE


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    if STATE.loaded:
        return HealthResponse(
            status="ok",
            artifacts_loaded=True,
            artifact_dir=STATE.artifact_dir,
            metadata_rows=STATE.metadata_rows,
            embeddings_shape=STATE.embeddings_shape,
            detail="Artifacts loaded and ready.",
        )
    return HealthResponse(
        status="degraded",
        artifacts_loaded=False,
        artifact_dir=STATE.artifact_dir,
        metadata_rows=STATE.metadata_rows,
        embeddings_shape=STATE.embeddings_shape,
        detail=STATE.load_error or "Artifacts not loaded.",
    )


@app.get("/api/v1/search", response_model=list[SearchItem], tags=["content"])
def search(
    q: str = Query(..., min_length=1, description="Title search text"),
    limit: int = Query(20, ge=1, le=100),
) -> list[SearchItem]:
    state = _require_loaded()
    df = state.work_df

    mask = df["title"].str.contains(q, case=False, regex=False)
    hits = df[mask].head(limit)

    return [
        SearchItem(
            id=int(idx),
            title=str(row["title"]),
            year=_extract_year(row.get("release_date")),
            rating=(
                round(float(row["vote_average"]), 1)
                if pd.notna(row.get("vote_average"))
                else None
            ),
            poster_url=_poster_url(row.get("poster_path")),
        )
        for idx, row in hits.iterrows()
    ]


@app.get("/api/v1/content/{id}", response_model=ContentDetail, tags=["content"])
def get_content(
    id: int = PathParam(..., ge=0, description="Movie index"),
) -> ContentDetail:
    state = _require_loaded()
    df = state.work_df

    if id < 0 or id >= len(df):
        raise HTTPException(status_code=404, detail=f"Movie with id={id} not found.")

    row = df.iloc[id]

    # Use pre-parsed display lists (original case)
    genres = state.display_genres[id]
    keywords = state.display_keywords[id]
    actors = state.display_actors[id]
    director = state.display_directors[id] or None
    collection_raw = str(row.get("collection_name", "")).strip()
    collection = collection_raw if collection_raw else None

    release_date_str = None
    rd = row.get("release_date")
    if pd.notna(rd):
        try:
            release_date_str = pd.Timestamp(rd).strftime("%Y-%m-%d")
        except Exception:
            release_date_str = str(rd)

    return ContentDetail(
        id=id,
        title=str(row["title"]),
        overview=str(row["overview"]) if row.get("overview") else None,
        genres=genres,
        keywords=keywords,
        actors=actors,
        director=director,
        collection=collection,
        vote_average=(
            round(float(row["vote_average"]), 1)
            if pd.notna(row.get("vote_average"))
            else None
        ),
        vote_count=(
            int(row["vote_count"]) if pd.notna(row.get("vote_count")) else None
        ),
        runtime=int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
        release_date=release_date_str,
        language=(
            str(row.get("original_language"))
            if pd.notna(row.get("original_language"))
            else None
        ),
        poster_url=_poster_url(row.get("poster_path")),
    )


@app.get(
    "/api/v1/recommend/{id}",
    response_model=list[RecommendationItem],
    tags=["recommendation"],
)
def recommend(
    id: int = PathParam(..., ge=0, description="Movie index"),
    limit: int = Query(10, ge=1, le=100),
) -> list[RecommendationItem]:
    state = _require_loaded()

    if id < 0 or id >= len(state.work_df):
        raise HTTPException(
            status_code=404, detail=f"Movie with id={id} not found."
        )

    results = recommend_by_index(
        work_df=state.work_df,
        embeddings=state.embeddings,
        runtime=state.runtime,
        q_idx=id,
        base_weights=state.base_weights,
        top_k=limit,
    )

    items: list[RecommendationItem] = []
    for _, row in results.iterrows():
        idx = int(row["index"])
        score = float(
            row.get(
                "rerank_score",
                row.get("quality_adjusted_score", row.get("final_score", 0)),
            )
        )
        items.append(
            RecommendationItem(
                id=idx,
                title=str(row["title"]),
                score=round(score, 4),
                poster_url=_poster_url(
                    state.work_df.iloc[idx].get("poster_path")
                ),
            )
        )
    return items


@app.get("/api/v1/genre/{genre}", response_model=list[SearchItem], tags=["browse"])
def by_genre(
    genre: str = PathParam(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: Literal["bayesian", "popularity"] = Query("bayesian"),
) -> list[SearchItem]:
    state = _require_loaded()
    df = state.work_df
    runtime = state.runtime

    genre_norm = genre.strip().lower()
    mask = pd.Series(
        [genre_norm in gs for gs in runtime["genre_sets"]],
        index=df.index,
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return []

    if sort_by == "bayesian":
        filtered["_sort"] = runtime["bayesian_norm_arr"][filtered.index]
    else:
        filtered["_sort"] = pd.to_numeric(
            filtered["vote_count"], errors="coerce"
        ).fillna(0)

    filtered = filtered.sort_values("_sort", ascending=False).head(limit)

    return [
        SearchItem(
            id=int(idx),
            title=str(row["title"]),
            year=_extract_year(row.get("release_date")),
            rating=(
                round(float(row["vote_average"]), 1)
                if pd.notna(row.get("vote_average"))
                else None
            ),
            poster_url=_poster_url(row.get("poster_path")),
        )
        for idx, row in filtered.iterrows()
    ]


@app.get("/api/v1/trending", response_model=list[SearchItem], tags=["browse"])
def trending(
    limit: int = Query(20, ge=1, le=100),
    sort_by: Literal["bayesian", "popularity"] = Query("popularity"),
) -> list[SearchItem]:
    state = _require_loaded()
    df = state.work_df.copy()
    runtime = state.runtime

    if sort_by == "bayesian":
        df["_sort"] = runtime["bayesian_norm_arr"]
    else:
        df["_sort"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)

    top = df.sort_values("_sort", ascending=False).head(limit)

    return [
        SearchItem(
            id=int(idx),
            title=str(row["title"]),
            year=_extract_year(row.get("release_date")),
            rating=(
                round(float(row["vote_average"]), 1)
                if pd.notna(row.get("vote_average"))
                else None
            ),
            poster_url=_poster_url(row.get("poster_path")),
        )
        for idx, row in top.iterrows()
    ]
