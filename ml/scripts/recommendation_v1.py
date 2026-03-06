from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_FEATURE_WEIGHTS = {
    "embedding": 0.33,
    "genre": 0.15,
    "keyword": 0.15,
    "actor": 0.10,
    "director": 0.05,
    "collection": 0.03,
    "bayesian": 0.15,
    "recency": 0.04,
}

RETRIEVAL_K_DEFAULT = 150
MIN_BAYES_SCORE_GATE = 0.60
MIN_BAYES_WITH_STRONG_MATCH = 0.55
MIN_STRONG_METADATA_GATE = 0.70
RECENCY_DECAY_YEARS = 12.0
TITLE_FAMILY_MATCH_THRESHOLD = 0.50
TITLE_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "movie",
    "film",
    "part",
    "chapter",
    "episode",
    "story",
    "stories",
    "adventure",
    "adventures",
    "return",
    "returns",
    "rise",
    "rises",
    "fall",
    "falls",
    "vs",
    "v",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}


def parse_list_field(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return [part.strip().lower() for part in text.split(",") if part.strip()]


def parse_set_field(value: Any) -> set[str]:
    return set(parse_list_field(value))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weight sum must be > 0")
    return {key: value / total for key, value in weights.items()}


def shift_weight(
    weights: dict[str, float],
    from_key: str,
    to_keys: list[str],
    amount: float,
) -> None:
    if amount <= 0 or not to_keys:
        return

    shift = min(float(amount), float(weights.get(from_key, 0.0)))
    if shift <= 0:
        return

    weights[from_key] -= shift
    per_target = shift / len(to_keys)
    for key in to_keys:
        weights[key] = weights.get(key, 0.0) + per_target


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def actor_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def tokenize_title(title: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and not token.isdigit() and token not in TITLE_STOPWORDS
    }


def title_family_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def load_metadata(artifact_dir: Path) -> pd.DataFrame:
    metadata_csv = artifact_dir / "movie_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Could not find metadata file in {artifact_dir} (expected movie_metadata.csv)."
        )
    return pd.read_csv(metadata_csv)


def load_artifacts(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    config_path = artifact_dir / "config.json"
    embeddings_path = artifact_dir / "overview_embeddings.npy"
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
        raise ValueError("feature_arrays.npz missing 'bayesian_scores_norm'.")

    bayesian_scores_norm = loaded_arrays["bayesian_scores_norm"].astype(np.float32)
    recency_scores = None
    if "recency_scores" in loaded_arrays.files:
        recency_scores = loaded_arrays["recency_scores"].astype(np.float32)

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} vs embeddings={embeddings.shape[0]}"
        )
    if len(metadata) != len(bayesian_scores_norm):
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} vs bayesian_scores_norm={len(bayesian_scores_norm)}"
        )
    if recency_scores is not None and len(metadata) != len(recency_scores):
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} vs recency_scores={len(recency_scores)}"
        )

    return metadata, embeddings, bayesian_scores_norm, recency_scores, config


def prepare_work_df(metadata: pd.DataFrame) -> pd.DataFrame:
    work_df = metadata.copy().reset_index(drop=True)

    work_df["title"] = work_df["title"].fillna("").astype(str)
    work_df["overview"] = work_df["overview"].fillna("").astype(str)
    if "collection_name" in work_df.columns:
        work_df["collection_name"] = work_df["collection_name"].fillna("").astype(str).str.strip().str.lower()
    else:
        work_df["collection_name"] = ""
    work_df["release_date"] = pd.to_datetime(work_df.get("release_date"), errors="coerce")

    work_df["_genre_set"] = work_df["genres"].apply(parse_set_field)
    work_df["_keyword_set"] = work_df["keywords"].apply(parse_set_field)
    work_df["_actor_list"] = work_df["top_5_actors"].apply(parse_list_field)
    work_df["_actor_set"] = work_df["_actor_list"].apply(set)
    work_df["_director_list"] = work_df["director"].apply(parse_list_field)
    work_df["_director_str"] = work_df["_director_list"].apply(lambda d: d[0] if d else "")
    work_df["_is_animation"] = work_df["_genre_set"].apply(lambda g: "animation" in g)

    work_df = work_df[work_df["title"].str.strip() != ""].reset_index(drop=True)
    return work_df


def build_runtime_data(
    work_df: pd.DataFrame,
    bayesian_scores_norm: np.ndarray,
    recency_scores: np.ndarray | None,
) -> dict[str, Any]:
    if recency_scores is None:
        current_timestamp = pd.Timestamp.utcnow().tz_localize(None)
        release_dates = work_df["release_date"]
        age_years = (
            (current_timestamp - release_dates).dt.days.fillna(365.25 * 100) / 365.25
        ).clip(lower=0.0)
        recency_scores = np.exp(-age_years / RECENCY_DECAY_YEARS).to_numpy(dtype=np.float32)

    return {
        "genre_sets": work_df["_genre_set"].tolist(),
        "keyword_sets": work_df["_keyword_set"].tolist(),
        "actor_sets": work_df["_actor_set"].tolist(),
        "titles_arr": work_df["title"].to_numpy(),
        "directors_arr": np.asarray(work_df["_director_str"].to_numpy(), dtype=object),
        "collections_arr": np.asarray(work_df["collection_name"].to_numpy(), dtype=object),
        "is_animation": work_df["_is_animation"].to_numpy(),
        "bayesian_norm_arr": np.asarray(bayesian_scores_norm, dtype=np.float32),
        "recency_scores_arr": np.asarray(recency_scores, dtype=np.float32),
        "title_token_sets": [tokenize_title(title) for title in work_df["title"].to_numpy()],
    }


def find_query_index(work_df: pd.DataFrame, name: str) -> int:
    name_norm = name.strip().lower()
    titles = work_df["title"].str.lower()

    exact = titles[titles == name_norm]
    if len(exact) > 0:
        return int(exact.index[0])

    contains = titles[titles.str.contains(name_norm, regex=False)]
    if len(contains) > 0:
        return int(contains.index[0])

    raise ValueError(f"Could not find movie with name: {name}")


def get_feature_weights(
    work_df: pd.DataFrame,
    runtime: dict[str, Any],
    q_idx: int,
    base_weights: dict[str, float],
) -> dict[str, float]:
    weights = base_weights.copy()

    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = runtime["directors_arr"][q_idx]
    q_collection = runtime["collections_arr"][q_idx]
    q_is_animation = bool(runtime["is_animation"][q_idx])
    q_release_date = work_df.iloc[q_idx]["release_date"]
    overview_word_count = len(str(work_df.iloc[q_idx]["overview"]).split())

    if q_is_animation:
        shift_weight(weights, "actor", ["genre", "keyword"], 0.06)

    if not q_director:
        shift_weight(weights, "director", ["embedding", "genre"], weights["director"])

    if not q_collection:
        shift_weight(weights, "collection", ["embedding", "genre"], weights["collection"])
    else:
        shift_weight(weights, "collection", ["genre", "keyword"], 0.02)

    if len(q_keywords) == 0:
        shift_weight(weights, "keyword", ["embedding", "genre"], weights["keyword"])
    elif len(q_keywords) < 3:
        shift_weight(weights, "keyword", ["embedding", "genre"], 0.05)
    elif len(q_keywords) >= 8:
        shift_weight(weights, "embedding", ["keyword"], 0.05)

    if len(q_actors) <= 1:
        shift_weight(weights, "actor", ["embedding", "genre"], weights["actor"] * 0.6)
    elif len(q_actors) >= 4 and not q_is_animation:
        shift_weight(weights, "embedding", ["actor"], 0.04)

    if len(q_genres) >= 3:
        shift_weight(weights, "embedding", ["genre"], 0.04)

    if overview_word_count < 25:
        shift_weight(weights, "embedding", ["genre", "keyword"], 0.07)

    if pd.isna(q_release_date):
        shift_weight(weights, "recency", ["embedding", "genre"], weights["recency"])

    return normalize_weights(weights)


def compute_quality_adjustment(
    embedding_scores: np.ndarray,
    genre_scores: np.ndarray,
    keyword_scores: np.ndarray,
    actor_scores: np.ndarray,
    director_scores: np.ndarray,
    collection_scores: np.ndarray,
    bayes_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    strong_metadata_match = np.maximum.reduce(
        [genre_scores, keyword_scores, actor_scores, director_scores, collection_scores]
    )

    quality_multiplier = 0.45 + 0.55 * bayes_scores

    low_quality_penalty = np.ones_like(bayes_scores, dtype=np.float32)
    low_quality_penalty = np.where(
        (bayes_scores < 0.55) & (strong_metadata_match < 0.55),
        0.75,
        low_quality_penalty,
    )
    low_quality_penalty = np.where(
        (bayes_scores < 0.45) & (strong_metadata_match < 0.50),
        0.60,
        low_quality_penalty,
    )
    low_quality_penalty = np.where(
        (bayes_scores < 0.35) & (strong_metadata_match < 0.35) & (embedding_scores > 0.60),
        0.45,
        low_quality_penalty,
    )

    rescue_bonus = np.ones_like(bayes_scores, dtype=np.float32)
    rescue_bonus = np.where((bayes_scores >= 0.40) & (director_scores > 0), 1.04, rescue_bonus)
    rescue_bonus = np.where(
        (bayes_scores >= 0.45) & (genre_scores > 0.60) & (keyword_scores > 0.30),
        1.03,
        rescue_bonus,
    )

    return quality_multiplier * low_quality_penalty * rescue_bonus, strong_metadata_match


def collection_cap_for_rank(rank_position: int) -> int:
    if rank_position < 10:
        return 2
    if rank_position < 20:
        return 3
    return 4


def director_cap_for_rank(rank_position: int) -> int:
    if rank_position < 10:
        return 2
    if rank_position < 20:
        return 3
    return 4


def title_family_cap_for_rank(rank_position: int) -> int:
    if rank_position < 10:
        return 2
    if rank_position < 20:
        return 3
    return 4


def structured_rerank(
    candidates_df: pd.DataFrame,
    embeddings: np.ndarray,
    runtime: dict[str, Any],
    q_idx: int,
    top_k: int,
) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df

    pool = candidates_df.copy().reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    selected_indices: list[int] = []
    collection_counts: dict[str, int] = {}
    director_counts: dict[str, int] = {}
    q_title_tokens = runtime["title_token_sets"][q_idx]

    while len(selected_rows) < min(top_k, len(pool)):
        rank_position = len(selected_rows)
        best_row_pos = None
        best_rerank_score = -np.inf

        for enforce_caps in (True, False):
            for row_pos in range(len(pool)):
                row = pool.iloc[row_pos]
                candidate_index = int(row["index"])
                candidate_collection = runtime["collections_arr"][candidate_index]
                candidate_director = runtime["directors_arr"][candidate_index]
                candidate_title_tokens = runtime["title_token_sets"][candidate_index]

                same_title_family_count = sum(
                    title_family_similarity(candidate_title_tokens, runtime["title_token_sets"][idx])
                    >= TITLE_FAMILY_MATCH_THRESHOLD
                    for idx in selected_indices
                )

                if enforce_caps:
                    if candidate_collection and collection_counts.get(candidate_collection, 0) >= collection_cap_for_rank(rank_position):
                        continue
                    if candidate_director and director_counts.get(candidate_director, 0) >= director_cap_for_rank(rank_position):
                        continue
                    if same_title_family_count >= title_family_cap_for_rank(rank_position):
                        continue

                query_title_family_score = title_family_similarity(q_title_tokens, candidate_title_tokens)
                collection_penalty = 0.12 * collection_counts.get(candidate_collection, 0) if candidate_collection else 0.0
                director_penalty = 0.05 * director_counts.get(candidate_director, 0) if candidate_director else 0.0
                title_family_penalty = 0.12 * same_title_family_count
                if query_title_family_score >= TITLE_FAMILY_MATCH_THRESHOLD:
                    title_family_penalty += 0.08

                if selected_indices:
                    max_genre_redundancy = max(
                        jaccard_similarity(runtime["genre_sets"][candidate_index], runtime["genre_sets"][idx])
                        for idx in selected_indices
                    )
                    max_keyword_redundancy = max(
                        jaccard_similarity(runtime["keyword_sets"][candidate_index], runtime["keyword_sets"][idx])
                        for idx in selected_indices
                    )
                    max_embedding_similarity = float(np.max(embeddings[selected_indices] @ embeddings[candidate_index]))
                else:
                    max_genre_redundancy = 0.0
                    max_keyword_redundancy = 0.0
                    max_embedding_similarity = 0.0

                extra_low_quality_penalty = 0.0
                if row["bayesian_rating_norm"] < 0.55 and row["strong_metadata_match"] < 0.55:
                    extra_low_quality_penalty = 0.07

                rerank_score = (
                    float(row["quality_adjusted_score"])
                    + 0.02 * float(row["recency_score"])
                    - collection_penalty
                    - director_penalty
                    - title_family_penalty
                    - 0.08 * max_genre_redundancy
                    - 0.10 * max_keyword_redundancy
                    - 0.08 * max_embedding_similarity
                    - extra_low_quality_penalty
                )

                if rerank_score > best_rerank_score:
                    best_rerank_score = rerank_score
                    best_row_pos = row_pos

            if best_row_pos is not None:
                break

        chosen = pool.iloc[best_row_pos].copy()
        chosen["rerank_score"] = best_rerank_score
        selected_rows.append(chosen)

        chosen_index = int(chosen["index"])
        selected_indices.append(chosen_index)

        chosen_collection = runtime["collections_arr"][chosen_index]
        chosen_director = runtime["directors_arr"][chosen_index]
        if candidate_collection := chosen_collection:
            collection_counts[candidate_collection] = collection_counts.get(candidate_collection, 0) + 1
        if candidate_director := chosen_director:
            director_counts[candidate_director] = director_counts.get(candidate_director, 0) + 1

        pool = pool.drop(index=best_row_pos).reset_index(drop=True)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def recommend_by_index(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    runtime: dict[str, Any],
    q_idx: int,
    base_weights: dict[str, float] | None = None,
    top_k: int = 10,
    retrieval_k: int = RETRIEVAL_K_DEFAULT,
) -> pd.DataFrame:
    n_total = len(work_df)
    if q_idx < 0 or q_idx >= n_total:
        raise ValueError(f"Invalid q_idx={q_idx}. Must be in [0, {n_total - 1}]")
    if n_total <= 1:
        return pd.DataFrame(columns=["index", "title", "final_score"])

    weights_base = normalize_weights((base_weights or BASE_FEATURE_WEIGHTS).copy())
    n_candidates = min(max(retrieval_k, top_k * 6), n_total - 1)

    query_vec = embeddings[q_idx]
    similarities = embeddings @ query_vec
    similarities[q_idx] = -1.0

    top_indices = np.argpartition(similarities, -n_candidates)[-n_candidates:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    candidate_indices = top_indices.astype(int)

    q_genres = runtime["genre_sets"][q_idx]
    q_keywords = runtime["keyword_sets"][q_idx]
    q_actors = runtime["actor_sets"][q_idx]
    q_director = runtime["directors_arr"][q_idx]
    q_collection = runtime["collections_arr"][q_idx]
    q_title_tokens = runtime["title_token_sets"][q_idx]

    weights = get_feature_weights(work_df, runtime, q_idx, weights_base)

    embedding_scores = similarities[candidate_indices].astype(np.float32)
    genre_scores = np.array(
        [jaccard_similarity(q_genres, runtime["genre_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    keyword_scores = np.array(
        [jaccard_similarity(q_keywords, runtime["keyword_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    actor_scores = np.array(
        [actor_overlap(q_actors, runtime["actor_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    director_scores = (
        (runtime["directors_arr"][candidate_indices] == q_director).astype(np.float32)
        if q_director
        else np.zeros(len(candidate_indices), dtype=np.float32)
    )
    collection_scores = (
        (runtime["collections_arr"][candidate_indices] == q_collection).astype(np.float32)
        if q_collection
        else np.zeros(len(candidate_indices), dtype=np.float32)
    )
    title_family_scores = np.array(
        [title_family_similarity(q_title_tokens, runtime["title_token_sets"][i]) for i in candidate_indices],
        dtype=np.float32,
    )
    bayes_scores = runtime["bayesian_norm_arr"][candidate_indices]
    recency_candidate_scores = runtime["recency_scores_arr"][candidate_indices]

    base_scores = (
        weights["embedding"] * embedding_scores
        + weights["genre"] * genre_scores
        + weights["keyword"] * keyword_scores
        + weights["actor"] * actor_scores
        + weights["director"] * director_scores
        + weights["collection"] * collection_scores
        + weights["bayesian"] * bayes_scores
        + weights["recency"] * recency_candidate_scores
    )

    quality_adjustment, strong_metadata_match = compute_quality_adjustment(
        embedding_scores=embedding_scores,
        genre_scores=genre_scores,
        keyword_scores=keyword_scores,
        actor_scores=actor_scores,
        director_scores=director_scores,
        collection_scores=collection_scores,
        bayes_scores=bayes_scores,
    )
    quality_adjusted_scores = base_scores * quality_adjustment

    base_ranked = pd.DataFrame(
        {
            "index": candidate_indices,
            "title": runtime["titles_arr"][candidate_indices],
            "final_score": base_scores,
            "quality_adjusted_score": quality_adjusted_scores,
            "embedding_score": embedding_scores,
            "genre_score": genre_scores,
            "keyword_score": keyword_scores,
            "actor_score": actor_scores,
            "director_match": director_scores.astype(int),
            "collection_match": collection_scores.astype(int),
            "title_family_score": title_family_scores,
            "bayesian_rating_norm": bayes_scores,
            "recency_score": recency_candidate_scores,
            "strong_metadata_match": strong_metadata_match,
        }
    ).sort_values("quality_adjusted_score", ascending=False).reset_index(drop=True)

    filtered_candidates = base_ranked[
        (base_ranked["bayesian_rating_norm"] >= MIN_BAYES_SCORE_GATE)
        | (
            (base_ranked["bayesian_rating_norm"] >= MIN_BAYES_WITH_STRONG_MATCH)
            & (base_ranked["strong_metadata_match"] >= MIN_STRONG_METADATA_GATE)
        )
    ].reset_index(drop=True)

    if len(filtered_candidates) < top_k:
        filtered_candidates = base_ranked.head(max(top_k * 3, top_k)).copy()

    return structured_rerank(
        filtered_candidates,
        embeddings=embeddings,
        runtime=runtime,
        q_idx=q_idx,
        top_k=top_k,
    )


def recommend_by_name(
    work_df: pd.DataFrame,
    embeddings: np.ndarray,
    runtime: dict[str, Any],
    name: str,
    base_weights: dict[str, float] | None = None,
    top_k: int = 10,
    retrieval_k: int = RETRIEVAL_K_DEFAULT,
) -> pd.DataFrame:
    q_idx = find_query_index(work_df, name=name)
    return recommend_by_index(
        work_df=work_df,
        embeddings=embeddings,
        runtime=runtime,
        q_idx=q_idx,
        base_weights=base_weights,
        top_k=top_k,
        retrieval_k=retrieval_k,
    )


def get_runtime_weights(config: dict[str, Any]) -> dict[str, float]:
    configured = config.get("weights", {})
    merged = BASE_FEATURE_WEIGHTS.copy()
    for key in merged:
        if key in configured:
            merged[key] = float(configured[key])
    return normalize_weights(merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run movie recommendation inference from CSV-only saved artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        default="ml/artifacts/content_recommendation_v2",
        help="Artifact directory containing config, movie_metadata.csv, embeddings, and feature arrays",
    )
    parser.add_argument("--query", default="The Dark Knight", help="Movie title to query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K_DEFAULT)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = project_root / artifact_dir

    metadata, embeddings, bayesian_scores_norm, recency_scores, config = load_artifacts(artifact_dir)
    work_df = prepare_work_df(metadata)
    runtime = build_runtime_data(work_df, bayesian_scores_norm, recency_scores)
    base_weights = get_runtime_weights(config)

    query_index = find_query_index(work_df, name=args.query)
    print("Query movie:", work_df.iloc[query_index]["title"])
    print("Dynamic weights:", get_feature_weights(work_df, runtime, query_index, base_weights))

    results = recommend_by_name(
        work_df=work_df,
        embeddings=embeddings,
        runtime=runtime,
        name=args.query,
        base_weights=base_weights,
        top_k=args.top_k,
        retrieval_k=args.retrieval_k,
    )

    display_cols = [
        "index",
        "title",
        "final_score",
        "quality_adjusted_score",
        "embedding_score",
        "genre_score",
        "keyword_score",
        "actor_score",
        "director_match",
        "collection_match",
        "title_family_score",
        "bayesian_rating_norm",
        "recency_score",
        "strong_metadata_match",
        "rerank_score",
    ]
    display_cols = [col for col in display_cols if col in results.columns]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
        print(results[display_cols])


if __name__ == "__main__":
    main()
