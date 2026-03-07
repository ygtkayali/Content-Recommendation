"""Two-stage movie recommendation: Stage 1 (content-based) → Stage 2 (LightGBM reranker).

Uses recommendation_v1 for Stage 1 candidate generation and applies a trained
LambdaRank model to rerank candidates based on richer feature signals learned
from MovieLens user ratings.

The reranker is a **global** model — no user features at inference time.

IMPORTANT: Feature computation must exactly match reranker_training.ipynb.
Training-specific helpers (_actor_overlap_weighted, _tokenize_simple,
_STAGE1_WEIGHTS) replicate the notebook's functions to avoid train/serve skew.
"""

from __future__ import annotations

import argparse
import json
import re as _re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommendation_v1 import (
    BASE_FEATURE_WEIGHTS,
    build_runtime_data,
    find_query_index,
    jaccard_similarity,
    load_artifacts,
    normalize_weights,
    prepare_work_df,
    recompute_item_feature_bayesian_norm,
)


# ---------------------------------------------------------------------------
# Training-consistent constants and helpers
# ---------------------------------------------------------------------------
# These MUST match reranker_training.ipynb exactly to avoid train/serve skew.

# Stage 1 weights used during training (excludes recency)
_STAGE1_WEIGHTS: dict[str, float] = {
    "embedding": 0.33,
    "genre": 0.15,
    "keyword": 0.15,
    "actor": 0.10,
    "director": 0.05,
    "collection": 0.03,
    "bayesian": 0.15,
}
_STAGE1_WEIGHT_SUM: float = sum(_STAGE1_WEIGHTS.values())  # 0.96

# Simple tokenizer matching training notebook (no stopword / length filtering)
_SIMPLE_TOKEN_RE = _re.compile(r"[a-z0-9]+")


def _tokenize_simple(title: str) -> set[str]:
    """Tokenize title the same way as training notebook's _tokenize_title."""
    return set(_SIMPLE_TOKEN_RE.findall(str(title).lower()))


def _actor_overlap_weighted(a_set: set[str], b_set: set[str]) -> float:
    """Position-weighted actor overlap matching training notebook's _actor_overlap.

    Converts sets to lists (arbitrary but deterministic per-run ordering),
    then computes a position-discounted overlap score.
    """
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


# ---------------------------------------------------------------------------
# Item features (from movie_user_data.ipynb)
# ---------------------------------------------------------------------------

def load_item_features(data_root: Path) -> pd.DataFrame:
    """Load item_features.csv/parquet produced by movie_user_data.ipynb."""
    item_features_csv = data_root / "processed" / "item_features.csv"
    item_features_parquet = data_root / "processed" / "item_features.parquet"

    if item_features_csv.exists():
        item_features = pd.read_csv(item_features_csv)
    elif item_features_parquet.exists():
        item_features = pd.read_parquet(item_features_parquet)
    else:
        raise FileNotFoundError(
            f"Missing item features at {item_features_csv} or {item_features_parquet}"
        )

    if "movie_id" not in item_features.columns:
        raise ValueError("item_features must include movie_id")

    item_features["movie_id"] = pd.to_numeric(item_features["movie_id"], errors="coerce")
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

    item_features = item_features.set_index("movie_id", drop=False)
    return recompute_item_feature_bayesian_norm(item_features)


# ---------------------------------------------------------------------------
# Reranker model artifacts
# ---------------------------------------------------------------------------

def load_reranker_artifacts(reranker_dir: Path):
    """Load LightGBM model + feature column spec from reranker directory.

    Returns (model, feature_columns) or (None, []) if artifacts are missing
    (graceful fallback).
    """
    model_path = reranker_dir / "model.txt"
    feature_cols_path = reranker_dir / "feature_columns.json"

    if not model_path.exists() or not feature_cols_path.exists():
        return None, []

    try:
        import lightgbm as lgb
    except ImportError:
        print("WARNING: lightgbm not installed — reranker disabled.")
        return None, []

    model = lgb.Booster(model_file=str(model_path))

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    feature_columns = payload.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "feature_columns.json must contain a non-empty list under 'feature_columns'."
        )

    return model, [str(col) for col in feature_columns]


# ---------------------------------------------------------------------------
# Stage 1: content-based candidate generation
# ---------------------------------------------------------------------------

def build_stage1_candidates(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    runtime: dict[str, Any],
    q_idx: int,
    embed_pool_size: int = 300,
    keep_top_n: int = 150,
) -> pd.DataFrame:
    """Generate Stage 1 candidates using the same scoring as training.

    Uses _STAGE1_WEIGHTS (no recency) and _actor_overlap_weighted to match
    the training notebook's Stage 1 simulation exactly.

    Returns a DataFrame with columns: index, title, stage1_score, and
    per-feature scores for diagnostic display.
    """
    n_total = len(work_df)
    if n_total <= 1:
        return pd.DataFrame(columns=["index", "title", "stage1_score"])

    embed_pool_size = max(1, min(embed_pool_size, n_total - 1))
    keep_top_n = max(1, min(keep_top_n, embed_pool_size))

    # Embedding retrieval
    query_vec = embeddings[q_idx]
    similarities = embeddings @ query_vec
    similarities[q_idx] = -1.0

    top_indices = np.argpartition(similarities, -embed_pool_size)[-embed_pool_size:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    candidate_indices = top_indices.astype(int)

    # Compute per-feature scores (matching training notebook)
    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = str(runtime["directors_arr"][q_idx])
    q_collection = str(runtime["collections_arr"][q_idx])

    emb_scores = similarities[candidate_indices].astype(np.float32)

    genre_scores = np.array(
        [jaccard_similarity(q_genres, runtime["genre_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    keyword_scores = np.array(
        [jaccard_similarity(q_keywords, runtime["keyword_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    # Use position-weighted actor overlap matching training
    actor_scores = np.array(
        [_actor_overlap_weighted(q_actors, runtime["actor_sets"][i]) for i in candidate_indices],
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

    bayes = runtime["bayesian_norm_arr"][candidate_indices]

    # Stage 1 weighted score — NO recency, divided by WEIGHT_SUM (matches training)
    weighted_total = (
        _STAGE1_WEIGHTS["embedding"] * emb_scores
        + _STAGE1_WEIGHTS["genre"] * genre_scores
        + _STAGE1_WEIGHTS["keyword"] * keyword_scores
        + _STAGE1_WEIGHTS["actor"] * actor_scores
        + _STAGE1_WEIGHTS["director"] * director_scores
        + _STAGE1_WEIGHTS["collection"] * collection_scores
        + _STAGE1_WEIGHTS["bayesian"] * bayes
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
            "bayesian_norm": bayes,
        }
    ).sort_values("stage1_score", ascending=False)

    return ranked.head(keep_top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: reranker feature construction
# ---------------------------------------------------------------------------

def build_reranker_features(
    candidates_df: pd.DataFrame,
    work_df: pd.DataFrame,
    runtime: dict[str, Any],
    q_idx: int,
    item_features: pd.DataFrame,
    reranker_feature_columns: list[str],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Build feature matrix aligned with reranker_feature_columns for each candidate.

    All feature computations match reranker_training.ipynb exactly:
    - _actor_overlap_weighted for actor_overlap
    - _tokenize_simple for title_token_overlap
    - _STAGE1_WEIGHTS / _STAGE1_WEIGHT_SUM for baseline_score
    - pandas .rank(method="average", pct=True) for similarity_percentile
    """
    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = str(runtime["directors_arr"][q_idx])
    q_collection = str(runtime["collections_arr"][q_idx])
    q_score = float(work_df.iloc[q_idx].get("vote_average", 0.0) or 0.0)
    q_title_tokens = _tokenize_simple(str(runtime["titles_arr"][q_idx]))

    # Precompute embedding rank via argsort (matches training)
    query_vec = embeddings[q_idx]
    sims_all = embeddings @ query_vec
    sims_all[q_idx] = -1.0
    rank_order = np.argsort(-sims_all)
    rank_lookup = np.empty_like(rank_order, dtype=np.int32)
    rank_lookup[rank_order] = np.arange(1, len(rank_order) + 1, dtype=np.int32)

    n_items_total = len(item_features)

    rows = []
    for _, row in candidates_df.iterrows():
        cand_idx = int(row["index"])

        # Get movie_id (TMDB ID) for item feature lookup
        cand_movie_id = pd.to_numeric(
            pd.Series([work_df.iloc[cand_idx].get("id")]), errors="coerce"
        ).iloc[0]
        cand_movie_id = int(cand_movie_id) if pd.notna(cand_movie_id) else -1

        cand_genres = runtime["genre_sets"][cand_idx]
        cand_keywords = runtime["keyword_sets"][cand_idx]
        cand_actors = runtime["actor_sets"][cand_idx]
        cand_director = str(runtime["directors_arr"][cand_idx])
        cand_collection = str(runtime["collections_arr"][cand_idx])
        cand_score = float(work_df.iloc[cand_idx].get("vote_average", 0.0) or 0.0)
        cand_title_tokens = _tokenize_simple(str(runtime["titles_arr"][cand_idx]))

        # Lookup item features — guard against duplicate index entries
        if cand_movie_id in item_features.index:
            _lookup = item_features.loc[cand_movie_id]
            item_feat = _lookup.iloc[0] if isinstance(_lookup, pd.DataFrame) else _lookup
        else:
            item_feat = None

        candidate_rating_count = float(item_feat["rating_count"]) if item_feat is not None else 0.0
        candidate_avg_rating = float(item_feat["avg_user_rating"]) if item_feat is not None else 0.0
        candidate_bayesian_norm = float(item_feat["bayesian_score_norm"]) if item_feat is not None else 0.0
        candidate_popularity_rank = float(item_feat["popularity_rank"]) if item_feat is not None else 0.0
        candidate_rating_std = (
            float(item_feat["rating_std"])
            if (item_feat is not None and "rating_std" in item_features.columns)
            else 0.0
        )

        emb_sim = float(row["embedding_score"])
        genre_ov = float(jaccard_similarity(q_genres, cand_genres))
        keyword_ov = float(jaccard_similarity(q_keywords, cand_keywords))
        actor_ov = float(_actor_overlap_weighted(q_actors, cand_actors))
        director_m = float(int(bool(q_director) and cand_director == q_director))
        collection_m = float(int(bool(q_collection) and cand_collection == q_collection))

        # baseline_score: must match training — uses _STAGE1_WEIGHTS / _STAGE1_WEIGHT_SUM
        baseline = (
            _STAGE1_WEIGHTS["embedding"] * emb_sim
            + _STAGE1_WEIGHTS["genre"] * genre_ov
            + _STAGE1_WEIGHTS["keyword"] * keyword_ov
            + _STAGE1_WEIGHTS["actor"] * actor_ov
            + _STAGE1_WEIGHTS["director"] * director_m
            + _STAGE1_WEIGHTS["collection"] * collection_m
            + _STAGE1_WEIGHTS["bayesian"] * candidate_bayesian_norm
        ) / _STAGE1_WEIGHT_SUM

        # title_token_overlap: uses _tokenize_simple + jaccard (matches training)
        title_overlap = float(jaccard_similarity(q_title_tokens, cand_title_tokens))

        feature_row = {
            "embedding_similarity": emb_sim,
            "genre_overlap": genre_ov,
            "keyword_overlap": keyword_ov,
            "actor_overlap": actor_ov,
            "director_match": director_m,
            "collection_match": collection_m,
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
            "baseline_score": baseline,
        }
        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)

    # similarity_percentile: use pandas .rank(method="average", pct=True) to match training
    if not feature_df.empty and "embedding_similarity" in feature_df.columns:
        feature_df["similarity_percentile"] = (
            feature_df["embedding_similarity"]
            .rank(method="average", pct=True)
            .astype(np.float32)
        )
    else:
        feature_df["similarity_percentile"] = 0.0

    # Ensure all expected columns exist (fill missing with 0)
    for col in reranker_feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df = feature_df[reranker_feature_columns].copy()
    for col in feature_df.columns:
        feature_df[col] = (
            pd.to_numeric(feature_df[col], errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )

    return feature_df


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) — post-reranker diversity
# ---------------------------------------------------------------------------

def mmr_rerank(
    candidates_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    lam: float = 0.7,
) -> pd.DataFrame:
    """Select top_k items from candidates_df using MMR.

    MMR score = λ · relevance  −  (1−λ) · max_sim_to_selected

    where relevance is the reranker_score (min–max normalised to [0,1])
    and similarity is the cosine similarity between candidate embeddings
    and already-selected embeddings.

    Parameters
    ----------
    candidates_df : DataFrame with columns 'index' (embedding row) and 'reranker_score'.
    embeddings    : (N, D) embedding matrix.
    top_k         : Number of items to select.
    lam           : Trade-off (1.0 = pure relevance, 0.0 = pure diversity).
    """
    if candidates_df.empty or top_k <= 0:
        return candidates_df.head(top_k).reset_index(drop=True)

    pool = candidates_df.copy().reset_index(drop=True)
    cand_indices = pool["index"].to_numpy(dtype=int)
    cand_embs = embeddings[cand_indices]  # (C, D)

    # Normalise reranker scores to [0, 1] for comparable λ weighting
    raw_scores = pool["reranker_score"].to_numpy(dtype=np.float64)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max - s_min > 1e-9:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(raw_scores)

    selected_order: list[int] = []  # row positions in pool
    remaining = set(range(len(pool)))

    for _ in range(min(top_k, len(pool))):
        best_pos = -1
        best_mmr = -np.inf

        if not selected_order:
            # First pick: highest relevance
            best_pos = int(np.argmax(norm_scores))
        else:
            sel_embs = cand_embs[selected_order]  # (S, D)
            # Similarity of every remaining candidate to every selected item
            sim_matrix = cand_embs @ sel_embs.T  # (C, S)
            max_sim = sim_matrix.max(axis=1)      # (C,)

            for pos in remaining:
                mmr_score = lam * norm_scores[pos] - (1.0 - lam) * max_sim[pos]
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_pos = pos

        selected_order.append(best_pos)
        remaining.discard(best_pos)

    result = pool.iloc[selected_order].copy().reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    artifact_dir: Path,
    reranker_dir: Path,
    data_root: Path,
    embed_pool_size: int = 300,
    stage1_top_n: int = 150,
    reranker_top_k: int = 40,
    final_top_k: int = 10,
    mmr_lambda: float = 0.7,
) -> pd.DataFrame:
    """Run the two-stage recommendation pipeline.

    Stage 1: Content-based candidate retrieval (matching training's Stage 1 simulation).
    Stage 2: LightGBM LambdaRank reranking using movie features + user-rating signals.

    Falls back to Stage 1 only if reranker model is not available.
    """
    metadata, embeddings, bayesian_scores_norm, recency_scores, config = load_artifacts(
        artifact_dir
    )
    work_df = prepare_work_df(metadata)
    runtime = build_runtime_data(work_df, bayesian_scores_norm, recency_scores)

    q_idx = find_query_index(work_df, name=query)

    # Stage 1: content-based candidate generation (reuses runtime)
    stage1_candidates = build_stage1_candidates(
        work_df=work_df,
        embeddings=embeddings,
        runtime=runtime,
        q_idx=q_idx,
        embed_pool_size=embed_pool_size,
        keep_top_n=stage1_top_n,
    )

    if stage1_candidates.empty:
        return stage1_candidates

    # Try loading reranker — graceful fallback if missing
    reranker_model, reranker_feature_columns = load_reranker_artifacts(reranker_dir)

    if reranker_model is None:
        print("INFO: Reranker model not found — returning Stage 1 results only.")
        result = stage1_candidates.head(final_top_k).reset_index(drop=True)
        result = result.rename(columns={"stage1_score": "final_score"})
        return result

    # Load item features for reranker quality signals
    item_features = load_item_features(data_root)

    # Stage 2: build features and rerank
    stage2_features = build_reranker_features(
        candidates_df=stage1_candidates,
        work_df=work_df,
        runtime=runtime,
        q_idx=q_idx,
        item_features=item_features,
        reranker_feature_columns=reranker_feature_columns,
        embeddings=embeddings,
    )

    reranker_scores = reranker_model.predict(
        stage2_features.to_numpy(dtype=np.float32)
    )

    reranked = stage1_candidates.copy()
    reranked["reranker_score"] = reranker_scores.astype(np.float32)
    reranked = (
        reranked.sort_values("reranker_score", ascending=False)
        .head(reranker_top_k)
        .reset_index(drop=True)
    )

    # Stage 3: MMR diversity selection from reranked shortlist
    final_df = mmr_rerank(
        candidates_df=reranked,
        embeddings=embeddings,
        top_k=final_top_k,
        lam=mmr_lambda,
    )
    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage movie recommendation: content-based retrieval + LightGBM reranker."
    )
    parser.add_argument("--query", default="Avatar", help="Movie title query")
    parser.add_argument(
        "--artifact-dir",
        default="ml/artifacts/content_recommendation_v2",
        help="Content-based artifact dir (metadata/embeddings/config/features)",
    )
    parser.add_argument(
        "--reranker-dir",
        default="ml/artifacts/content_recommendation_v2/reranker_v2",
        help="Reranker artifact dir (model.txt, feature_columns.json)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root (expects processed/item_features.*)",
    )
    parser.add_argument(
        "--embed-pool-size",
        type=int,
        default=300,
        help="Stage 1 embedding retrieval pool size (matches training STAGE1_RETRIEVAL_POOL)",
    )
    parser.add_argument(
        "--stage1-top-n",
        type=int,
        default=150,
        help="Stage 1 top-N after multi-feature scoring (matches training STAGE1_CANDIDATES_PER_QUERY)",
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=40,
        help="Stage 2 reranked shortlist size",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Final returned recommendations",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda (1.0 = pure relevance, 0.0 = pure diversity)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = project_root / artifact_dir

    reranker_dir = Path(args.reranker_dir)
    if not reranker_dir.is_absolute():
        reranker_dir = project_root / reranker_dir

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    final_df = run_pipeline(
        query=args.query,
        artifact_dir=artifact_dir,
        reranker_dir=reranker_dir,
        data_root=data_root,
        embed_pool_size=args.embed_pool_size,
        stage1_top_n=args.stage1_top_n,
        reranker_top_k=args.reranker_top_k,
        final_top_k=args.final_top_k,
        mmr_lambda=args.mmr_lambda,
    )

    if final_df.empty:
        print("No recommendations produced.")
        return

    print(f"Query: {args.query}")
    print(
        "Pipeline: Stage1(embedding retrieval → multi-feature scoring) "
        f"→ Stage2(LightGBM reranker v2) → MMR(λ={args.mmr_lambda})"
    )

    display_cols = [
        "index",
        "title",
        "stage1_score",
        "reranker_score",
        "embedding_score",
        "genre_score",
        "keyword_score",
        "actor_score",
        "director_match",
        "collection_match",
        "bayesian_norm",
    ]
    display_cols = [col for col in display_cols if col in final_df.columns]

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 180
    ):
        print(final_df[display_cols].reset_index(drop=True))


if __name__ == "__main__":
    main()
