"""
InboxPilot — Utility helpers.

Text normalization, keyword matching, and common helpers used across
the environment, graders, and reward engine.
"""

from __future__ import annotations

import re
from typing import Iterable


def normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_keywords(text: str, keywords: Iterable[str], threshold: int = 1) -> bool:
    """Return True if *text* contains at least *threshold* keywords."""
    norm = normalize(text)
    count = sum(1 for kw in keywords if kw.lower() in norm)
    return count >= threshold


def keyword_match_ratio(text: str, keywords: list[str]) -> float:
    """Return fraction of *keywords* found in *text*."""
    if not keywords:
        return 1.0
    norm = normalize(text)
    matched = sum(1 for kw in keywords if kw.lower() in norm)
    return matched / len(keywords)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))
