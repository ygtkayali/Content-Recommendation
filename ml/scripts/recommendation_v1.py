from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_WEIGHTS = {
    "synopsis": 0.45,
    "genre": 0.20,
    "type": 0.10,
    "studio": 0.05,
    "source": 0.05,
    "bayesian": 0.15,
}

weight_sum = sum(FEATURE_WEIGHTS.values())
if weight_sum <= 0:
    raise ValueError("Sum of feature weights must be > 0")


def parse_genre_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()

    text = str(value).strip()
    if not text:
        return set()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return {str(item).strip().lower() for item in parsed if str(item).strip()}
        except (ValueError, SyntaxError):
            pass

    parts = [part.strip().lower() for part in text.split(",")]
    return {part for part in parts if part}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)


def load_metadata(artifact_dir: Path) -> pd.DataFrame:
    metadata_csv = artifact_dir / "anime_metadata.csv"
    metadata_parquet = artifact_dir / "anime_metadata.parquet"

    if metadata_csv.exists():
        return pd.read_csv(metadata_csv)
    if metadata_parquet.exists():
        return pd.read_parquet(metadata_parquet)

    raise FileNotFoundError(
        f"Could not find metadata file in {artifact_dir} (expected anime_metadata.csv or anime_metadata.parquet)."
    )


def load_artifacts(artifact_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    config_path = artifact_dir / "config.json"
    embeddings_path = artifact_dir / "synopsis_embeddings.npy"
    feature_arrays_path = artifact_dir / "feature_arrays.npz"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing artifact: {config_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing artifact: {embeddings_path}")
    if not feature_arrays_path.exists():
        raise FileNotFoundError(f"Missing artifact: {feature_arrays_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    metadata = load_metadata(artifact_dir)
    embeddings = np.load(embeddings_path).astype(np.float32)
    loaded_arrays = np.load(feature_arrays_path)

    if "bayesian_scores_norm" not in loaded_arrays.files:
        raise ValueError("feature_arrays.npz does not contain 'bayesian_scores_norm'.")

    bayesian_scores_norm = loaded_arrays["bayesian_scores_norm"].astype(np.float32)

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} vs embeddings={embeddings.shape[0]}"
        )

    if len(metadata) != len(bayesian_scores_norm):
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} vs bayesian_scores_norm={len(bayesian_scores_norm)}"
        )

    expected_weights = config.get("weights", {})
    if expected_weights:
        for key, value in FEATURE_WEIGHTS.items():
            cfg_value = expected_weights.get(key)
            if cfg_value is not None and not np.isclose(float(cfg_value), float(value)):
                print(
                    f"Warning: config weight for '{key}' is {cfg_value}, script uses {value}."
                )

    return metadata, embeddings, bayesian_scores_norm


def find_query_index(work_df: pd.DataFrame, title_col: str, name: str) -> int:
    name_norm = name.strip().lower()
    titles = work_df[title_col].fillna("").astype(str).str.lower()

    exact = titles[titles == name_norm]
    if len(exact) > 0:
        return int(exact.index[0])

    contains = titles[titles.str.contains(name_norm, regex=False)]
    if len(contains) > 0:
        return int(contains.index[0])

    raise ValueError(f"Could not find anime with name: {name}")


def apply_mmr(
    candidates_df: pd.DataFrame,
    embeddings_matrix: np.ndarray,
    top_k: int,
    lambda_mmr: float = 0.7,
) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df

    if not (0.0 <= lambda_mmr <= 1.0):
        raise ValueError("lambda_mmr must be between 0 and 1")

    pool = candidates_df.copy().reset_index(drop=True)
    selected_rows = []
    selected_indices = []

    while len(selected_rows) < min(top_k, len(pool)):
        best_row_pos = None
        best_mmr_score = -np.inf

        for row_pos in range(len(pool)):
            candidate_index = int(pool.iloc[row_pos]["index"])
            relevance = float(pool.iloc[row_pos]["final_score"])

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                selected_matrix = embeddings_matrix[selected_indices]
                candidate_vec = embeddings_matrix[candidate_index]
                similarities = selected_matrix @ candidate_vec
                diversity_penalty = float(np.max(similarities))

            mmr_score = lambda_mmr * relevance - (1.0 - lambda_mmr) * diversity_penalty

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_row_pos = row_pos

        chosen = pool.iloc[best_row_pos].copy()
        chosen["mmr_score"] = best_mmr_score
        selected_rows.append(chosen)
        selected_indices.append(int(chosen["index"]))
        pool = pool.drop(index=best_row_pos).reset_index(drop=True)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def recommend_by_index(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    title_col: str,
    genres_col: str,
    type_col: str,
    studios_col: str,
    source_col: str,
    bayesian_scores_norm: np.ndarray,
    q_idx: int,
    top_k: int = 10,
    top_k_mmr: int = 40,
    lambda_mmr: float = 0.7,
) -> pd.DataFrame:
    n_total = len(work_df)
    if q_idx < 0 or q_idx >= n_total:
        raise ValueError(f"Invalid q_idx={q_idx}. Must be in [0, {n_total - 1}]")

    if n_total <= 1:
        return pd.DataFrame(columns=["index", "title", "final_score"])
    if top_k_mmr < top_k:
        top_k_mmr = top_k

    genre_sets = work_df["_genre_set"].tolist()
    types = work_df[type_col].fillna("").astype(str).str.strip().str.lower().tolist()
    studios = work_df[studios_col].fillna("").astype(str).str.strip().str.lower().tolist()
    sources = work_df[source_col].fillna("").astype(str).str.strip().str.lower().tolist()

    titles_arr = work_df[title_col].fillna("").astype(str).to_numpy()
    types_arr = np.asarray(types, dtype=object)
    studios_arr = np.asarray(studios, dtype=object)
    sources_arr = np.asarray(sources, dtype=object)
    bayesian_scores_norm_arr = np.asarray(bayesian_scores_norm, dtype=np.float32)

    n_candidates = min(top_k_mmr, n_total - 1)

    query_vec = embeddings[q_idx]
    similarities = embeddings @ query_vec
    similarities[q_idx] = -1.0

    top_indices = np.argpartition(similarities, -n_candidates)[-n_candidates:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    q_genres = genre_sets[q_idx]
    q_type = types_arr[q_idx]
    q_studio = studios_arr[q_idx]
    q_source = sources_arr[q_idx]

    candidate_indices = top_indices.astype(int)
    synopsis_scores = similarities[candidate_indices].astype(np.float32)
    genre_scores = np.array(
        [jaccard_similarity(q_genres, genre_sets[i]) for i in candidate_indices], dtype=np.float32
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

    base_ranked = pd.DataFrame(
        {
            "index": candidate_indices,
            "title": titles_arr[candidate_indices],
            "final_score": final_scores,
            "synopsis_score": synopsis_scores,
            "genre_score": genre_scores,
            "type_match": type_scores.astype(int),
            "studio_match": studio_scores.astype(int),
            "source_match": source_scores.astype(int),
            "bayesian_rating_norm": bayes_scores_norm,
        }
    ).sort_values("final_score", ascending=False).reset_index(drop=True)

    mmr_pool = base_ranked.head(top_k_mmr).reset_index(drop=True)
    mmr_ranked = apply_mmr(
        mmr_pool,
        embeddings_matrix=embeddings,
        top_k=top_k,
        lambda_mmr=lambda_mmr,
    )
    return mmr_ranked


def recommend_by_name(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    title_col: str,
    genres_col: str,
    type_col: str,
    studios_col: str,
    source_col: str,
    bayesian_scores_norm: np.ndarray,
    name: str,
    top_k: int = 10,
    top_k_mmr: int = 40,
    lambda_mmr: float = 0.7,
) -> pd.DataFrame:
    q_idx = find_query_index(work_df, title_col=title_col, name=name)
    return recommend_by_index(
        work_df=work_df,
        embeddings=embeddings,
        title_col=title_col,
        genres_col=genres_col,
        type_col=type_col,
        studios_col=studios_col,
        source_col=source_col,
        bayesian_scores_norm=bayesian_scores_norm,
        q_idx=q_idx,
        top_k=top_k,
        top_k_mmr=top_k_mmr,
        lambda_mmr=lambda_mmr,
    )


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of {candidates} found. Available columns: {list(df.columns)}")


def prepare_work_df(metadata: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str, str, str]:
    title_col = pick_column(metadata, ["Name", "name", "Title", "title"])
    genres_col = pick_column(metadata, ["Genres", "genres", "Genre", "genre", "genre_list"])
    type_col = pick_column(metadata, ["Type", "type"])
    studios_col = pick_column(metadata, ["Studios", "studios", "Studio", "studio"])
    source_col = pick_column(metadata, ["Source", "source"])

    work_df = metadata.copy().reset_index(drop=True)
    work_df[title_col] = work_df[title_col].fillna("").astype(str)
    work_df["_genre_set"] = work_df[genres_col].apply(parse_genre_set)

    return work_df, title_col, genres_col, type_col, studios_col, source_col


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run recommendation demo from saved artifacts (no embedding rebuild)."
    )
    parser.add_argument(
        "--artifact-dir",
        default="ml/artifacts/content_based_anime_v1",
        help="Artifact directory containing config/embeddings/metadata/features",
    )
    parser.add_argument("--query", default="Death Note", help="Anime title to query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-k-mmr", type=int, default=50)
    parser.add_argument("--lambda-mmr", type=float, default=0.6)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = project_root / artifact_dir

    metadata, embeddings, bayesian_scores_norm = load_artifacts(artifact_dir)
    work_df, title_col, genres_col, type_col, studios_col, source_col = prepare_work_df(metadata)

    query_index = find_query_index(work_df, title_col=title_col, name=args.query)
    print("Query anime:", work_df.iloc[query_index][title_col])

    demo_results = recommend_by_name(
        work_df=work_df,
        embeddings=embeddings,
        title_col=title_col,
        genres_col=genres_col,
        type_col=type_col,
        studios_col=studios_col,
        source_col=source_col,
        bayesian_scores_norm=bayesian_scores_norm,
        name=args.query,
        top_k=args.top_k,
        top_k_mmr=args.top_k_mmr,
        lambda_mmr=args.lambda_mmr,
    )

    display_cols = [
        "index",
        "title",
        "final_score",
        "synopsis_score",
        "genre_score",
        "type_match",
        "studio_match",
        "source_match",
        "bayesian_rating_norm",
        "mmr_score",
    ]
    display_cols = [col for col in display_cols if col in demo_results.columns]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(demo_results[display_cols])


if __name__ == "__main__":
    main()
