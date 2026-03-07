from __future__ import annotations

import ast
import json
import os
import re as _re
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
    jaccard_similarity,
    load_artifacts,
    prepare_work_df,
    recommend_by_index,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
DEFAULT_DATA_ROOT = "data"
DEFAULT_RERANKER_SUBDIR = "reranker_v2"
DEFAULT_STAGE1_RETRIEVAL_POOL = 300
DEFAULT_STAGE1_TOP_N = 150
DEFAULT_RERANKER_TOP_K = 40
DEFAULT_MMR_LAMBDA = 0.7

# Training-consistent reranker helpers.
_STAGE1_WEIGHTS: dict[str, float] = {
    "embedding": 0.33,
    "genre": 0.15,
    "keyword": 0.15,
    "actor": 0.10,
    "director": 0.05,
    "collection": 0.03,
    "bayesian": 0.15,
}
_STAGE1_WEIGHT_SUM: float = sum(_STAGE1_WEIGHTS.values())
_SIMPLE_TOKEN_RE = _re.compile(r"[a-z0-9]+")


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
    reranker_loaded: bool
    artifact_dir: str
    reranker_dir: str
    data_root: str
    metadata_rows: int
    item_features_rows: int
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
    item_features: pd.DataFrame | None = None
    reranker_model: Any | None = None
    reranker_feature_columns: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    base_weights: dict[str, float] = field(
        default_factory=lambda: BASE_FEATURE_WEIGHTS.copy()
    )
    loaded: bool = False
    reranker_loaded: bool = False
    load_error: str | None = None
    reranker_error: str | None = None
    artifact_dir: str = ""
    reranker_dir: str = ""
    data_root: str = ""

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

    @property
    def item_features_rows(self) -> int:
        return len(self.item_features) if self.item_features is not None else 0


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


def _get_data_root() -> Path:
    return _resolve_runtime_path(
        os.getenv("DATA_ROOT", DEFAULT_DATA_ROOT),
        PROJECT_ROOT,
    )


def _get_reranker_dir(artifact_dir: Path) -> Path:
    return artifact_dir / os.getenv("RERANKER_SUBDIR", DEFAULT_RERANKER_SUBDIR)


def _get_allowed_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    return origins or ["http://localhost:3000"]


def _tokenize_simple(title: str) -> set[str]:
    return set(_SIMPLE_TOKEN_RE.findall(str(title).lower()))


def _actor_overlap_weighted(a_set: set[str], b_set: set[str]) -> float:
    if not a_set or not b_set:
        return 0.0
    a_list, b_list = list(a_set), list(b_set)
    score = 0.0
    for i, actor in enumerate(a_list):
        if actor in b_set:
            j = b_list.index(actor) if actor in b_list else len(b_list)
            score += (1.0 / (i + 1)) * (1.0 / (j + 1))
    max_possible = sum(
        1.0 / (k + 1) ** 2 for k in range(min(len(a_list), len(b_list)))
    )
    return score / max_possible if max_possible > 0 else 0.0


def _load_item_features(data_root: Path) -> pd.DataFrame:
    item_features_csv = data_root / "processed" / "item_features.csv"

    if not item_features_csv.exists():
        raise FileNotFoundError(
            f"Missing item features at {item_features_csv}"
        )

    item_features = pd.read_csv(item_features_csv)

    if "movie_id" not in item_features.columns:
        raise ValueError("item_features must include movie_id")

    item_features["movie_id"] = pd.to_numeric(
        item_features["movie_id"], errors="coerce"
    )
    item_features = item_features.dropna(subset=["movie_id"]).copy()
    item_features["movie_id"] = item_features["movie_id"].astype("int64")

    expected_cols = [
        "avg_user_rating",
        "rating_count",
        "rating_std",
        "positive_ratio",
        "bayesian_score_norm",
        "popularity_rank",
    ]
    for col in expected_cols:
        if col not in item_features.columns:
            item_features[col] = 0.0

    return item_features.set_index("movie_id", drop=False)


def _load_reranker_artifacts(reranker_dir: Path) -> tuple[Any, list[str]]:
    model_path = reranker_dir / "model.txt"
    feature_cols_path = reranker_dir / "feature_columns.json"

    if not model_path.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(
            f"Missing reranker artifacts in {reranker_dir}. Expected model.txt and feature_columns.json."
        )

    import lightgbm as lgb

    model = lgb.Booster(model_file=str(model_path))

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    feature_columns = payload.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "feature_columns.json must contain a non-empty list under 'feature_columns'."
        )

    return model, [str(col) for col in feature_columns]


def _build_stage1_candidates(
    state: AppState,
    q_idx: int,
    embed_pool_size: int,
    keep_top_n: int,
) -> pd.DataFrame:
    if state.work_df is None or state.embeddings is None or state.runtime is None:
        return pd.DataFrame(columns=["index", "title", "stage1_score"])

    n_total = len(state.work_df)
    if n_total <= 1:
        return pd.DataFrame(columns=["index", "title", "stage1_score"])

    embed_pool_size = max(1, min(embed_pool_size, n_total - 1))
    keep_top_n = max(1, min(keep_top_n, embed_pool_size))

    runtime = state.runtime
    embeddings = state.embeddings

    query_vec = embeddings[q_idx]
    similarities = embeddings @ query_vec
    similarities[q_idx] = -1.0

    top_indices = np.argpartition(similarities, -embed_pool_size)[-embed_pool_size:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    candidate_indices = top_indices.astype(int)

    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = str(runtime["directors_arr"][q_idx])
    q_collection = str(runtime["collections_arr"][q_idx])

    emb_scores = similarities[candidate_indices].astype(np.float32)
    genre_scores = np.array(
        [
            jaccard_similarity(q_genres, runtime["genre_sets"][i])
            for i in candidate_indices
        ],
        dtype=np.float32,
    )
    keyword_scores = np.array(
        [
            jaccard_similarity(q_keywords, runtime["keyword_sets"][i])
            for i in candidate_indices
        ],
        dtype=np.float32,
    )
    actor_scores = np.array(
        [
            _actor_overlap_weighted(q_actors, runtime["actor_sets"][i])
            for i in candidate_indices
        ],
        dtype=np.float32,
    )
    director_scores = np.array(
        [
            1.0
            if q_director and str(runtime["directors_arr"][i]) == q_director
            else 0.0
            for i in candidate_indices
        ],
        dtype=np.float32,
    )
    collection_scores = np.array(
        [
            1.0
            if q_collection and str(runtime["collections_arr"][i]) == q_collection
            else 0.0
            for i in candidate_indices
        ],
        dtype=np.float32,
    )
    bayes_scores = runtime["bayesian_norm_arr"][candidate_indices]

    weighted_total = (
        _STAGE1_WEIGHTS["embedding"] * emb_scores
        + _STAGE1_WEIGHTS["genre"] * genre_scores
        + _STAGE1_WEIGHTS["keyword"] * keyword_scores
        + _STAGE1_WEIGHTS["actor"] * actor_scores
        + _STAGE1_WEIGHTS["director"] * director_scores
        + _STAGE1_WEIGHTS["collection"] * collection_scores
        + _STAGE1_WEIGHTS["bayesian"] * bayes_scores
    ) / _STAGE1_WEIGHT_SUM

    ranked = pd.DataFrame(
        {
            "index": candidate_indices,
            "title": runtime["titles_arr"][candidate_indices],
            "stage1_score": weighted_total,
            "embedding_score": emb_scores,
            "genre_score": genre_scores,
            "keyword_score": keyword_scores,
            "actor_score": actor_scores,
            "director_match": director_scores.astype(int),
            "collection_match": collection_scores.astype(int),
            "bayesian_norm": bayes_scores,
        }
    ).sort_values("stage1_score", ascending=False)

    return ranked.head(keep_top_n).reset_index(drop=True)


def _build_reranker_features(
    state: AppState,
    candidates_df: pd.DataFrame,
    q_idx: int,
) -> pd.DataFrame:
    if (
        state.work_df is None
        or state.runtime is None
        or state.item_features is None
        or state.embeddings is None
    ):
        return pd.DataFrame(columns=state.reranker_feature_columns)

    work_df = state.work_df
    runtime = state.runtime
    item_features = state.item_features
    embeddings = state.embeddings

    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = str(runtime["directors_arr"][q_idx])
    q_collection = str(runtime["collections_arr"][q_idx])
    q_score = float(work_df.iloc[q_idx].get("vote_average", 0.0) or 0.0)
    q_title_tokens = _tokenize_simple(str(runtime["titles_arr"][q_idx]))

    query_vec = embeddings[q_idx]
    sims_all = embeddings @ query_vec
    sims_all[q_idx] = -1.0
    rank_order = np.argsort(-sims_all)
    rank_lookup = np.empty_like(rank_order, dtype=np.int32)
    rank_lookup[rank_order] = np.arange(1, len(rank_order) + 1, dtype=np.int32)

    n_items_total = len(item_features)
    rows: list[dict[str, float | int]] = []

    for _, row in candidates_df.iterrows():
        cand_idx = int(row["index"])
        cand_movie_id_raw = pd.to_numeric(
            work_df.iloc[cand_idx].get("id"), errors="coerce"
        )
        cand_movie_id = (
            int(cand_movie_id_raw) if pd.notna(cand_movie_id_raw) else -1
        )

        cand_genres = runtime["genre_sets"][cand_idx]
        cand_keywords = runtime["keyword_sets"][cand_idx]
        cand_actors = runtime["actor_sets"][cand_idx]
        cand_director = str(runtime["directors_arr"][cand_idx])
        cand_collection = str(runtime["collections_arr"][cand_idx])
        cand_score = float(work_df.iloc[cand_idx].get("vote_average", 0.0) or 0.0)
        cand_title_tokens = _tokenize_simple(str(runtime["titles_arr"][cand_idx]))

        if cand_movie_id in item_features.index:
            lookup = item_features.loc[cand_movie_id]
            item_feat = lookup.iloc[0] if isinstance(lookup, pd.DataFrame) else lookup
        else:
            item_feat = None

        candidate_rating_count = (
            float(item_feat["rating_count"]) if item_feat is not None else 0.0
        )
        candidate_avg_rating = (
            float(item_feat["avg_user_rating"]) if item_feat is not None else 0.0
        )
        candidate_bayesian_norm = (
            float(item_feat["bayesian_score_norm"]) if item_feat is not None else 0.0
        )
        candidate_popularity_rank = (
            float(item_feat["popularity_rank"]) if item_feat is not None else 0.0
        )
        candidate_rating_std = (
            float(item_feat["rating_std"]) if item_feat is not None else 0.0
        )

        emb_sim = float(row["embedding_score"])
        genre_overlap = float(jaccard_similarity(q_genres, cand_genres))
        keyword_overlap = float(jaccard_similarity(q_keywords, cand_keywords))
        actor_overlap = float(_actor_overlap_weighted(q_actors, cand_actors))
        director_match = float(int(bool(q_director) and cand_director == q_director))
        collection_match = float(
            int(bool(q_collection) and cand_collection == q_collection)
        )
        baseline_score = (
            _STAGE1_WEIGHTS["embedding"] * emb_sim
            + _STAGE1_WEIGHTS["genre"] * genre_overlap
            + _STAGE1_WEIGHTS["keyword"] * keyword_overlap
            + _STAGE1_WEIGHTS["actor"] * actor_overlap
            + _STAGE1_WEIGHTS["director"] * director_match
            + _STAGE1_WEIGHTS["collection"] * collection_match
            + _STAGE1_WEIGHTS["bayesian"] * candidate_bayesian_norm
        ) / _STAGE1_WEIGHT_SUM

        title_overlap = float(jaccard_similarity(q_title_tokens, cand_title_tokens))

        rows.append(
            {
                "embedding_similarity": emb_sim,
                "genre_overlap": genre_overlap,
                "keyword_overlap": keyword_overlap,
                "actor_overlap": actor_overlap,
                "director_match": director_match,
                "collection_match": collection_match,
                "candidate_avg_rating": candidate_avg_rating,
                "candidate_rating_count": candidate_rating_count,
                "candidate_positive_ratio": (
                    float(item_feat["positive_ratio"]) if item_feat is not None else 0.0
                ),
                "candidate_bayesian_norm": candidate_bayesian_norm,
                "candidate_popularity_rank": candidate_popularity_rank,
                "rating_count_log": float(np.log1p(max(candidate_rating_count, 0.0))),
                "score_diff": float(abs(q_score - cand_score)),
                "candidate_popularity_percentile": (
                    float(1.0 - candidate_popularity_rank / n_items_total)
                    if n_items_total > 0
                    else 0.0
                ),
                "embedding_rank": int(rank_lookup[cand_idx]),
                "title_token_overlap": title_overlap,
                "genre_overlap_count": float(len(q_genres & cand_genres)),
                "keyword_overlap_count": float(len(q_keywords & cand_keywords)),
                "candidate_rating_cv": (
                    float(candidate_rating_std / candidate_avg_rating)
                    if candidate_avg_rating > 0.01
                    else 0.0
                ),
                "baseline_score": baseline_score,
            }
        )

    feature_df = pd.DataFrame(rows)

    if not feature_df.empty and "embedding_similarity" in feature_df.columns:
        feature_df["similarity_percentile"] = (
            feature_df["embedding_similarity"]
            .rank(method="average", pct=True)
            .astype(np.float32)
        )
    else:
        feature_df["similarity_percentile"] = 0.0

    for col in state.reranker_feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df = feature_df[state.reranker_feature_columns].copy()
    for col in feature_df.columns:
        feature_df[col] = (
            pd.to_numeric(feature_df[col], errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )

    return feature_df


def _mmr_rerank(
    candidates_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    lam: float = DEFAULT_MMR_LAMBDA,
) -> pd.DataFrame:
    if candidates_df.empty or top_k <= 0:
        return candidates_df.head(top_k).reset_index(drop=True)

    pool = candidates_df.copy().reset_index(drop=True)
    cand_indices = pool["index"].to_numpy(dtype=int)
    cand_embs = embeddings[cand_indices]

    raw_scores = pool["reranker_score"].to_numpy(dtype=np.float64)
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max - score_min > 1e-9:
        norm_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(raw_scores)

    selected_order: list[int] = []
    remaining = set(range(len(pool)))

    for _ in range(min(top_k, len(pool))):
        best_pos = -1
        best_mmr = -np.inf

        if not selected_order:
            best_pos = int(np.argmax(norm_scores))
        else:
            sel_embs = cand_embs[selected_order]
            sim_matrix = cand_embs @ sel_embs.T
            max_sim = sim_matrix.max(axis=1)

            for pos in remaining:
                mmr_score = lam * norm_scores[pos] - (1.0 - lam) * max_sim[pos]
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_pos = pos

        selected_order.append(best_pos)
        remaining.discard(best_pos)

    return pool.iloc[selected_order].copy().reset_index(drop=True)


def _recommend_with_reranker(
    state: AppState,
    q_idx: int,
    limit: int,
) -> pd.DataFrame:
    if (
        state.work_df is None
        or state.embeddings is None
        or state.runtime is None
        or state.reranker_model is None
        or state.item_features is None
        or not state.reranker_feature_columns
    ):
        raise ValueError("Reranker state is not fully loaded.")

    embed_pool_size = max(DEFAULT_STAGE1_RETRIEVAL_POOL, limit * 20)
    stage1_top_n = max(DEFAULT_STAGE1_TOP_N, limit * 10)
    reranker_top_k = max(DEFAULT_RERANKER_TOP_K, limit * 4)

    stage1_candidates = _build_stage1_candidates(
        state=state,
        q_idx=q_idx,
        embed_pool_size=embed_pool_size,
        keep_top_n=stage1_top_n,
    )
    if stage1_candidates.empty:
        return stage1_candidates

    stage2_features = _build_reranker_features(
        state=state,
        candidates_df=stage1_candidates,
        q_idx=q_idx,
    )
    reranker_scores = state.reranker_model.predict(
        stage2_features.to_numpy(dtype=np.float32)
    )

    reranked = stage1_candidates.copy()
    reranked["reranker_score"] = np.asarray(reranker_scores, dtype=np.float32)
    reranked = (
        reranked.sort_values("reranker_score", ascending=False)
        .head(reranker_top_k)
        .reset_index(drop=True)
    )

    return _mmr_rerank(
        candidates_df=reranked,
        embeddings=state.embeddings,
        top_k=limit,
        lam=DEFAULT_MMR_LAMBDA,
    )


def _load_app_state() -> AppState:
    """Load artifacts once and prepare all runtime data."""
    state = AppState()
    artifact_dir = _get_artifact_dir()
    data_root = _get_data_root()
    reranker_dir = _get_reranker_dir(artifact_dir)
    state.artifact_dir = str(artifact_dir)
    state.data_root = str(data_root)
    state.reranker_dir = str(reranker_dir)

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

        try:
            state.item_features = _load_item_features(data_root)
            state.reranker_model, state.reranker_feature_columns = (
                _load_reranker_artifacts(reranker_dir)
            )
            state.reranker_loaded = True
        except Exception as exc:
            state.reranker_loaded = False
            state.reranker_error = str(exc)
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
        detail = "Artifacts loaded and ready."
        if not STATE.reranker_loaded:
            detail = (
                "Base artifacts loaded, but reranker is unavailable. "
                f"{STATE.reranker_error or 'No details available.'}"
            )
        return HealthResponse(
            status="ok",
            artifacts_loaded=True,
            reranker_loaded=STATE.reranker_loaded,
            artifact_dir=STATE.artifact_dir,
            reranker_dir=STATE.reranker_dir,
            data_root=STATE.data_root,
            metadata_rows=STATE.metadata_rows,
            item_features_rows=STATE.item_features_rows,
            embeddings_shape=STATE.embeddings_shape,
            detail=detail,
        )
    return HealthResponse(
        status="degraded",
        artifacts_loaded=False,
        reranker_loaded=False,
        artifact_dir=STATE.artifact_dir,
        reranker_dir=STATE.reranker_dir,
        data_root=STATE.data_root,
        metadata_rows=STATE.metadata_rows,
        item_features_rows=STATE.item_features_rows,
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

    if state.reranker_loaded:
        results = _recommend_with_reranker(state=state, q_idx=id, limit=limit)
    else:
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
