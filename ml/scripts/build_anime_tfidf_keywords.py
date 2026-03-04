from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ml.src.text_features import join_keywords, normalize_text


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None


def extract_noun_phrases(text):
    if nlp is None:
        return []
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]


def noun_phrase_word_sets(noun_phrases: Iterable[str]) -> list[set[str]]:
    phrase_sets: list[set[str]] = []
    for phrase in noun_phrases:
        normalized = normalize_text(phrase)
        words = {word for word in normalized.split() if word}
        if words:
            phrase_sets.append(words)
    return phrase_sets


def candidate_in_noun_phrases(candidate: str, phrase_sets: list[set[str]]) -> bool:
    candidate_words = {word for word in normalize_text(candidate).split() if word}
    if not candidate_words:
        return False
    return any(candidate_words.issubset(phrase_words) for phrase_words in phrase_sets)


def resolve_synopsis_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    fallback_candidates = ["Synopsis", "Synposis", "summary", "description"]
    for candidate in fallback_candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find synopsis column. Requested '{requested}', available columns: {list(df.columns)}"
    )


def resolve_title_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    fallback_candidates = ["Name", "name", "Title", "title"]
    for candidate in fallback_candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find title column. Requested '{requested}', available columns: {list(df.columns)}"
    )


def build_custom_stopwords() -> set[str]:
    domain_noise = {
        "anime",
        "episode",
        "episodes",
        "story",
        "character",
        "characters",
        "guy",
        "girl",
        "boy",
        "man",
        "woman",
        "young",
        "day",
        "days",
        "night",
        "friend",
        "friends",
        "school",
        "life",
        "new",
        "meet",
        "meets",
        "met",
        "things",
        "thing",
        "way",
        "world",
        "time",
    }
    return set(ENGLISH_STOP_WORDS).union(domain_noise)


def token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]{3,}", normalize_text(str(text)))
    return set(tokens)


def filter_keywords(
    candidates: list[str],
    title_tokens: set[str],
    custom_stopwords: set[str],
    max_terms: int,
) -> list[str]:
    filtered: list[str] = []
    for term in candidates:
        term_norm = term.strip().lower()
        if not term_norm:
            continue

        words = term_norm.split()
        if any(word in custom_stopwords for word in words):
            continue

        if len(words) == 1:
            single = words[0]
            if single in title_tokens:
                continue
            if len(single) < 4:
                continue

        if term_norm in filtered:
            continue

        filtered.append(term_norm)
        if len(filtered) >= max_terms:
            break

    return filtered


def build_keywords(
    df: pd.DataFrame,
    synopsis_col: str,
    title_col: str,
    max_features: int,
    top_n: int,
    min_df: int,
    max_df: float,
) -> pd.DataFrame:
    work_df = df.copy()
    work_df[synopsis_col] = work_df[synopsis_col].fillna("").astype(str).map(normalize_text)
    work_df[title_col] = work_df[title_col].fillna("").astype(str)

    noun_phrase_sets_per_row = [
        noun_phrase_word_sets(extract_noun_phrases(text)) for text in work_df[synopsis_col]
    ]

    custom_stopwords = build_custom_stopwords()

    vectorizer = TfidfVectorizer(
        stop_words=list(custom_stopwords),
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"\b[a-zA-Z]{4,}\b",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(work_df[synopsis_col])
    vocab = vectorizer.get_feature_names_out()

    def top_keywords(row_index: int) -> list[str]:
        row = tfidf_matrix.getrow(row_index)
        if row.nnz == 0:
            return []
        indices = row.indices
        scores = row.data
        ranked = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
        candidates = [vocab[idx] for idx, _ in ranked]
        noun_phrase_sets = noun_phrase_sets_per_row[row_index]
        if noun_phrase_sets:
            candidates = [term for term in candidates if candidate_in_noun_phrases(term, noun_phrase_sets)]
        title_tokens = token_set(work_df.iloc[row_index][title_col])
        return filter_keywords(candidates, title_tokens, custom_stopwords, top_n)

    keyword_lists = [top_keywords(i) for i in range(tfidf_matrix.shape[0])]
    work_df["keywords_tfidf"] = keyword_lists
    work_df["keywords_tfidf_text"] = [join_keywords(items) for items in keyword_lists]

    return work_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TF-IDF keywords from anime synopsis text.")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path")
    parser.add_argument("--output", required=True, help="Output path (.csv or .parquet)")
    parser.add_argument("--synopsis-col", default="Synopsis", help="Synopsis column name")
    parser.add_argument("--title-col", default="Name", help="Title column name")
    parser.add_argument("--max-features", type=int, default=12000, help="Max TF-IDF features")
    parser.add_argument("--top-n", type=int, default=15, help="Top keywords per row")
    parser.add_argument("--min-df", type=int, default=3, help="Minimum document frequency")
    parser.add_argument("--max-df", type=float, default=0.35, help="Maximum document frequency ratio")

    args = parser.parse_args()

    if nlp is None:
        print(
            "Warning: spaCy model 'en_core_web_sm' not found. "
            "Proceeding without noun-phrase filtering (TF-IDF fallback mode)."
        )
        print(
            "Install with: python -m spacy download en_core_web_sm"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    synopsis_col = resolve_synopsis_column(df, args.synopsis_col)
    title_col = resolve_title_column(df, args.title_col)
    result_df = build_keywords(
        df=df,
        synopsis_col=synopsis_col,
        title_col=title_col,
        max_features=args.max_features,
        top_n=args.top_n,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        result_df.to_parquet(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False)

    print(f"Saved TF-IDF keyword output to: {output_path}")


if __name__ == "__main__":
    main()
