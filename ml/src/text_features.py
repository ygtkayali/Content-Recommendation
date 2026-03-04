from __future__ import annotations

import re
from typing import Iterable


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def join_keywords(tokens: Iterable[str]) -> str:
    return ", ".join(tokens)
