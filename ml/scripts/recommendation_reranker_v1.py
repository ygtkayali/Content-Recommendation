from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from recommendation_v1 import (
    FEATURE_WEIGHTS,
    apply_mmr,
    find_query_index,
    jaccard_similarity,
    load_artifacts,
    parse_genre_set,
    pick_column,
    prepare_work_df,
    weight_sum,
)


def load_item_features(data_root: Path) -> pd.DataFrame:
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

    if "anime_id" not in item_features.columns:
        raise ValueError("item_features must include anime_id")

    item_features["anime_id"] = pd.to_numeric(item_features["anime_id"], errors="coerce")
    item_features = item_features.dropna(subset=["anime_id"]).copy()
    item_features["anime_id"] = item_features["anime_id"].astype("int64")

    expected_cols = [
        "avg_user_rating",
        "rating_count",
        "positive_ratio",
        "bayesian_score_norm",
        "popularity_rank",
    ]
    for col in expected_cols:
        if col not in item_features.columns:
            item_features[col] = 0.0

    return item_features.set_index("anime_id", drop=False)


def load_reranker_artifacts(reranker_dir: Path) -> tuple[lgb.Booster, list[str]]:
    model_path = reranker_dir / "model.txt"
    feature_cols_path = reranker_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing reranker model: {model_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing reranker feature spec: {feature_cols_path}")

    model = lgb.Booster(model_file=str(model_path))

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    feature_columns = payload.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("feature_columns.json must contain a non-empty list under 'feature_columns'.")

    return model, [str(col) for col in feature_columns]


def build_stage1_candidates(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    bayesian_scores_norm: np.ndarray,
    title_col: str,
    type_col: str,
    studios_col: str,
    source_col: str,
    q_idx: int,
    embed_pool_size: int,
    keep_top_n: int,
) -> pd.DataFrame:
    n_total = len(work_df)
    if n_total <= 1:
        return pd.DataFrame(columns=["index", "title", "final_score"])

    if q_idx < 0 or q_idx >= n_total:
        raise ValueError(f"Invalid q_idx={q_idx}. Must be in [0, {n_total - 1}]")

    embed_pool_size = max(1, min(embed_pool_size, n_total - 1))
    keep_top_n = max(1, min(keep_top_n, embed_pool_size))

    genre_sets = work_df["_genre_set"].tolist()
    types = work_df[type_col].fillna("").astype(str).str.strip().str.lower().tolist()
    studios = work_df[studios_col].fillna("").astype(str).str.strip().str.lower().tolist()
    sources = work_df[source_col].fillna("").astype(str).str.strip().str.lower().tolist()

    titles_arr = work_df[title_col].fillna("").astype(str).to_numpy()
    types_arr = np.asarray(types, dtype=object)
    studios_arr = np.asarray(studios, dtype=object)
    sources_arr = np.asarray(sources, dtype=object)
    bayesian_scores_norm_arr = np.asarray(bayesian_scores_norm, dtype=np.float32)

    query_vec = embeddings[q_idx]
    similarities = embeddings @ query_vec
    similarities[q_idx] = -1.0

    top_indices = np.argpartition(similarities, -embed_pool_size)[-embed_pool_size:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    q_genres = genre_sets[q_idx]
    q_type = types_arr[q_idx]
    q_studio = studios_arr[q_idx]
    q_source = sources_arr[q_idx]

    candidate_indices = top_indices.astype(int)
    synopsis_scores = similarities[candidate_indices].astype(np.float32)
    genre_scores = np.array(
        [jaccard_similarity(q_genres, genre_sets[i]) for i in candidate_indices],
        dtype=np.float32,
    )

    if q_type:
        type_scores = (types_arr[candidate_indices] == q_type).astype(np.float32)
    else:
        type_scores = np.zeros(len(candidate_indices), dtype=np.float32)

    if q_studio:
        studio_scores = (studios_arr[candidate_indices] == q_studio).astype(np.float32)
    else:
        studio_scores = np.zeros(len(candidate_indices), dtype=np.float32)

    if q_source:
        source_scores = (sources_arr[candidate_indices] == q_source).astype(np.float32)
    else:
        source_scores = np.zeros(len(candidate_indices), dtype=np.float32)

    bayes_scores_norm = bayesian_scores_norm_arr[candidate_indices]

    weighted_total = (
        FEATURE_WEIGHTS["synopsis"] * synopsis_scores
        + FEATURE_WEIGHTS["genre"] * genre_scores
        + FEATURE_WEIGHTS["type"] * type_scores
        + FEATURE_WEIGHTS["studio"] * studio_scores
        + FEATURE_WEIGHTS["source"] * source_scores
        + FEATURE_WEIGHTS["bayesian"] * bayes_scores_norm
    )
    final_scores = weighted_total / weight_sum

    ranked = pd.DataFrame(
        {
            "index": candidate_indices,
            "title": titles_arr[candidate_indices],
            "stage1_score": final_scores,
            "synopsis_score": synopsis_scores,
            "genre_score": genre_scores,
            "type_match": type_scores.astype(int),
            "studio_match": studio_scores.astype(int),
            "source_match": source_scores.astype(int),
            "bayesian_rating_norm": bayes_scores_norm,
        }
    ).sort_values("stage1_score", ascending=False)

    return ranked.head(keep_top_n).reset_index(drop=True)


def _tokenize_title(title: str) -> set[str]:
    """Split title into lowercase alphanumeric tokens for overlap computation."""
    import re
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def build_reranker_features(
    candidates_df: pd.DataFrame,
    work_df: pd.DataFrame,
    q_idx: int,
    item_features: pd.DataFrame,
    reranker_feature_columns: list[str],
    embeddings: np.ndarray,
    title_col: str,
    type_col: str,
    studios_col: str,
    source_col: str,
    score_col: str,
) -> pd.DataFrame:
    q_genres = work_df.iloc[q_idx]["_genre_set"]
    q_type = str(work_df.iloc[q_idx][type_col]).strip().lower()
    q_studio = str(work_df.iloc[q_idx][studios_col]).strip().lower()
    q_source = str(work_df.iloc[q_idx][source_col]).strip().lower()
    q_score = pd.to_numeric(pd.Series([work_df.iloc[q_idx][score_col]]), errors="coerce").fillna(0.0).iloc[0]
    q_title_tokens = _tokenize_title(str(work_df.iloc[q_idx][title_col]))

    # Precompute embedding rank for this query
    query_vec = embeddings[q_idx]
    sims_all = embeddings @ query_vec
    sims_all[q_idx] = -1.0
    rank_order = np.argsort(-sims_all)
    rank_lookup = np.empty_like(rank_order)
    rank_lookup[rank_order] = np.arange(1, len(rank_order) + 1)

    n_items_total = len(item_features)

    rows = []
    for _, row in candidates_df.iterrows():
        cand_idx = int(row["index"])
        cand_row = work_df.iloc[cand_idx]

        cand_anime_id = pd.to_numeric(pd.Series([cand_row.get("anime_id")]), errors="coerce").iloc[0]
        cand_anime_id = int(cand_anime_id) if pd.notna(cand_anime_id) else -1

        cand_genres = cand_row["_genre_set"]
        cand_type = str(cand_row[type_col]).strip().lower()
        cand_studio = str(cand_row[studios_col]).strip().lower()
        cand_source = str(cand_row[source_col]).strip().lower()
        cand_score = pd.to_numeric(pd.Series([cand_row[score_col]]), errors="coerce").fillna(0.0).iloc[0]
        cand_title_tokens = _tokenize_title(str(cand_row[title_col]))

        item_feat = item_features.loc[cand_anime_id] if cand_anime_id in item_features.index else None

        candidate_rating_count = float(item_feat["rating_count"]) if item_feat is not None else 0.0
        candidate_avg_rating = float(item_feat["avg_user_rating"]) if item_feat is not None else 0.0
        candidate_bayesian_norm = float(item_feat["bayesian_score_norm"]) if item_feat is not None else 0.0
        candidate_popularity_rank = float(item_feat["popularity_rank"]) if item_feat is not None else 0.0
        candidate_rating_std = float(item_feat["rating_std"]) if (item_feat is not None and "rating_std" in item_features.columns) else 0.0

        emb_sim = float(row["synopsis_score"])
        genre_ov = float(jaccard_similarity(q_genres, cand_genres))
        type_m = float(int(bool(q_type) and cand_type == q_type))
        studio_m = float(int(bool(q_studio) and cand_studio == q_studio))
        source_m = float(int(bool(q_source) and cand_source == q_source))

        # Baseline score: Stage 1 multi-feature weighted score
        baseline = (
            FEATURE_WEIGHTS["synopsis"] * emb_sim
            + FEATURE_WEIGHTS["genre"] * genre_ov
            + FEATURE_WEIGHTS["type"] * type_m
            + FEATURE_WEIGHTS["studio"] * studio_m
            + FEATURE_WEIGHTS["source"] * source_m
            + FEATURE_WEIGHTS["bayesian"] * candidate_bayesian_norm
        ) / weight_sum

        # Title token overlap (franchise detection)
        title_overlap = jaccard_similarity(q_title_tokens, cand_title_tokens)

        feature_row = {
            "embedding_similarity": emb_sim,
            "genre_overlap": genre_ov,
            "type_match": type_m,
            "studio_match": studio_m,
            "source_match": source_m,
            "candidate_avg_rating": candidate_avg_rating,
            "candidate_rating_count": candidate_rating_count,
            "candidate_positive_ratio": float(item_feat["positive_ratio"]) if item_feat is not None else 0.0,
            "candidate_bayesian_norm": candidate_bayesian_norm,
            "candidate_popularity_rank": candidate_popularity_rank,
            "rating_count_log": float(np.log1p(max(candidate_rating_count, 0.0))),
            "score_diff": float(abs(float(q_score) - float(cand_score))),
            # New features
            "candidate_popularity_percentile": float(1.0 - candidate_popularity_rank / n_items_total) if n_items_total > 0 else 0.0,
            "embedding_rank": int(rank_lookup[cand_idx]),
            "metadata_match_count": int(type_m + studio_m + source_m),
            "title_token_overlap": title_overlap,
            "genre_overlap_count": float(len(q_genres & cand_genres)),
            "candidate_rating_cv": float(candidate_rating_std / candidate_avg_rating) if candidate_avg_rating > 0.01 else 0.0,
            "baseline_score": baseline,
        }

        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)

    # Compute similarity_percentile within this candidate pool
    if not feature_df.empty and "embedding_similarity" in feature_df.columns:
        sims = feature_df["embedding_similarity"].to_numpy(dtype=np.float64)
        n = len(sims)
        if n <= 1:
            feature_df["similarity_percentile"] = 1.0
        else:
            order = np.argsort(sims)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            feature_df["similarity_percentile"] = (ranks / (n - 1)).astype(np.float32)
    else:
        feature_df["similarity_percentile"] = 0.0

    for col in reranker_feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df = feature_df[reranker_feature_columns].copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).astype(np.float32)

    return feature_df


def apply_mmr_on_reranked(
    reranked_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    lambda_mmr: float,
) -> pd.DataFrame:
    if reranked_df.empty:
        return reranked_df

    tmp = reranked_df.copy()
    tmp["final_score"] = tmp["reranker_score"].astype(float)

    mmr_df = apply_mmr(
        candidates_df=tmp,
        embeddings_matrix=embeddings,
        top_k=top_k,
        lambda_mmr=lambda_mmr,
    )

    mmr_df = mmr_df.rename(columns={"mmr_score": "post_mmr_score"})
    return mmr_df.drop(columns=["final_score"], errors="ignore")


def run_pipeline(
    query: str,
    artifact_dir: Path,
    reranker_dir: Path,
    data_root: Path,
    embed_pool_size: int,
    stage1_top_n: int,
    reranker_top_k: int,
    final_top_k: int,
    lambda_mmr: float,
    use_mmr: bool,
) -> pd.DataFrame:
    metadata, embeddings, bayesian_scores_norm = load_artifacts(artifact_dir)
    work_df, title_col, _, type_col, studios_col, source_col = prepare_work_df(metadata)

    if "anime_id" not in work_df.columns:
        raise ValueError("Metadata must include anime_id for reranker item feature lookup.")

    score_col = pick_column(work_df, ["Score", "score", "rating", "Rating"])

    item_features = load_item_features(data_root)
    reranker_model, reranker_feature_columns = load_reranker_artifacts(reranker_dir)

    q_idx = find_query_index(work_df, title_col=title_col, name=query)

    stage1_candidates = build_stage1_candidates(
        work_df=work_df,
        embeddings=embeddings,
        bayesian_scores_norm=bayesian_scores_norm,
        title_col=title_col,
        type_col=type_col,
        studios_col=studios_col,
        source_col=source_col,
        q_idx=q_idx,
        embed_pool_size=embed_pool_size,
        keep_top_n=stage1_top_n,
    )

    if stage1_candidates.empty:
        return stage1_candidates

    stage2_features = build_reranker_features(
        candidates_df=stage1_candidates,
        work_df=work_df,
        q_idx=q_idx,
        item_features=item_features,
        reranker_feature_columns=reranker_feature_columns,
        embeddings=embeddings,
        title_col=title_col,
        type_col=type_col,
        studios_col=studios_col,
        source_col=source_col,
        score_col=score_col,
    )

    reranker_scores = reranker_model.predict(stage2_features.to_numpy(dtype=np.float32))

    reranked = stage1_candidates.copy()
    reranked["reranker_score"] = reranker_scores.astype(np.float32)
    reranked = reranked.sort_values("reranker_score", ascending=False).head(reranker_top_k).reset_index(drop=True)

    if use_mmr:
        final_df = apply_mmr_on_reranked(
            reranked_df=reranked,
            embeddings=embeddings,
            top_k=final_top_k,
            lambda_mmr=lambda_mmr,
        )
    else:
        final_df = reranked.head(final_top_k).reset_index(drop=True)

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage recommendation with LightGBM reranker (candidate gen + rerank + optional MMR)."
    )
    parser.add_argument("--query", default="Death Note", help="Anime title query")
    parser.add_argument(
        "--artifact-dir",
        default="ml/artifacts/content_based_anime_v1",
        help="Content-based artifact dir (metadata/embeddings/config/features)",
    )
    parser.add_argument(
        "--reranker-dir",
        default="ml/artifacts/content_based_anime_v1/reranker_v2",
        help="Reranker artifact dir (model.txt, feature_columns.json)",
    )
    parser.add_argument("--data-root", default="data", help="Data root (expects processed/item_features.*)")
    parser.add_argument("--embed-pool-size", type=int, default=600, help="Stage 1 embedding retrieval pool size")
    parser.add_argument("--stage1-top-n", type=int, default=120, help="Stage 1 top-N after multi-feature scoring")
    parser.add_argument("--reranker-top-k", type=int, default=40, help="Stage 3 reranked shortlist size")
    parser.add_argument("--final-top-k", type=int, default=10, help="Final returned recommendations")
    parser.add_argument("--lambda-mmr", type=float, default=0.7, help="MMR lambda in [0,1]")
    parser.add_argument(
        "--disable-mmr",
        action="store_true",
        help="Disable Stage 4 post-processing MMR",
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
        lambda_mmr=args.lambda_mmr,
        use_mmr=not args.disable_mmr,
    )

    if final_df.empty:
        print("No recommendations produced.")
        return

    print(f"Query: {args.query}")
    print(
        "Pipeline: Stage1(embed->scoring) -> Stage2(features) -> Stage3(rerank)"
        + (" -> Stage4(MMR)" if not args.disable_mmr else "")
    )

    display_cols = [
        "index",
        "title",
        "stage1_score",
        "reranker_score",
        "post_mmr_score",
        "synopsis_score",
        "genre_score",
        "type_match",
        "studio_match",
        "source_match",
        "bayesian_rating_norm",
    ]
    display_cols = [col for col in display_cols if col in final_df.columns]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
        print(final_df[display_cols].reset_index(drop=True))


if __name__ == "__main__":
    main()
