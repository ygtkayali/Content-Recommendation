from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_column(df: pd.DataFrame, requested: str, fallbacks: list[str]) -> str:
    if requested in df.columns:
        return requested
    for candidate in fallbacks:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find column. Requested '{requested}', available columns: {list(df.columns)}"
    )


def truncate_text(value: str, max_len: int) -> str:
    value = str(value).strip().replace("\n", " ")
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TF-IDF keywords with random sample inspection.")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path with generated keywords")
    parser.add_argument("--samples", type=int, default=20, help="Number of random samples to display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--title-col", default="Name", help="Title column name")
    parser.add_argument("--synopsis-col", default="Synopsis", help="Synopsis column name")
    parser.add_argument("--keywords-col", default="keywords_tfidf_text", help="Generated keywords column")
    parser.add_argument("--max-synopsis-len", type=int, default=300, help="Max synopsis chars to print")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    title_col = resolve_column(df, args.title_col, ["title", "Title", "name", "Name"])
    synopsis_col = resolve_column(df, args.synopsis_col, ["Synposis", "summary", "description", "Description"])
    keywords_col = resolve_column(df, args.keywords_col, ["keywords_tfidf", "keywords", "Keywords"])

    if df.empty:
        print("Dataset is empty.")
        return

    sample_size = min(args.samples, len(df))
    sampled = df.sample(n=sample_size, random_state=args.seed)

    print(f"Showing {sample_size} random samples from {input_path}")
    print(f"Using columns => title: '{title_col}', synopsis: '{synopsis_col}', keywords: '{keywords_col}'")
    print("=" * 100)

    for idx, row in sampled.iterrows():
        title = truncate_text(row.get(title_col, ""), 120)
        synopsis = truncate_text(row.get(synopsis_col, ""), args.max_synopsis_len)
        keywords = row.get(keywords_col, "")
        if isinstance(keywords, list):
            keywords_text = ", ".join(str(item) for item in keywords)
        else:
            keywords_text = str(keywords)

        print(f"Row Index: {idx}")
        print(f"Title: {title}")
        print(f"Synopsis: {synopsis}")
        print(f"Keywords: {keywords_text}")
        print("-" * 100)


if __name__ == "__main__":
    main()
