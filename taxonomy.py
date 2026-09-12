"""
Narrows the Google Product Taxonomy down to a shortlist.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

CATEGORIES_FILE = Path(__file__).parent / "categories.txt"

DEFAULT_CANDIDATE_COUNT = 30
# Used by the repair call, alongside model-supplied keywords.
WIDE_CANDIDATE_COUNT = 100

# How much description text feeds the retrieval query.
DESCRIPTION_QUERY_CHARS = 120

# How many visible-text blocks to fall back on when a page declares nothing structured. 
FALLBACK_BLOCKS = 5

# Tokens that carry no discriminative signal in a taxonomy of product categories.
_STOPWORDS = {"and", "or", "the", "a", "an", "of", "for", "with", "in", "other", "&"}

# Leaf segment matches count for more than ancestor matches: a query word matching
# "Drills" is far stronger evidence than one matching "Hardware".
_LEAF_WEIGHT = 2.5


def _tokenise(text: str) -> list[str]:
    # Split text into lowercase word tokens, dropping stopwords.
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        tokens.append(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)
    return tokens


@lru_cache(maxsize=1)
def _index() -> tuple[list[str], list[dict[str, float]], dict[str, float]]:
    # Build the search index once and cache it for the process.
    categories: list[str] = []
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                categories.append(line)

    weights: list[dict[str, float]] = []
    document_frequency: dict[str, int] = {}

    for category in categories:
        segments = [segment.strip() for segment in category.split(">")]
        leaf_tokens = set(_tokenise(segments[-1]))

        token_weights: dict[str, float] = {}
        for token in _tokenise(category):
            weight = _LEAF_WEIGHT if token in leaf_tokens else 1.0
            # A token repeated across segments should not compound; keep the strongest.
            token_weights[token] = max(token_weights.get(token, 0.0), weight)

        weights.append(token_weights)
        for token in token_weights:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    total = len(categories)
    idf = {
        token: math.log(total / (1 + count)) for token, count in document_frequency.items()
    }

    return categories, weights, idf


def build_query(
    name: str | None = None,
    brand: str | None = None,
    description: str | None = None,
    breadcrumbs: list[str] | None = None,
    declared_category: str | None = None,
    fallback_text: list[str] | None = None,
) -> str:
    # Build the text we search the taxonomy with.

    parts: list[str] = []

    if breadcrumbs:
        # Drop the first crumb: it is nearly always the site name, not a category.
        meaningful = breadcrumbs[1:] if len(breadcrumbs) > 1 else breadcrumbs
        parts.extend(meaningful)
        parts.extend(meaningful)  # repeated to weight it

    if declared_category:
        parts.append(declared_category)
        parts.append(declared_category)

    if name:
        parts.append(name)
    if description:
        # A hard character cap rather than "the first sentence".
        parts.append(description.strip()[:DESCRIPTION_QUERY_CHARS])

    # Only reached when nothing structured was available.
    if not parts and fallback_text:
        for block in fallback_text[:FALLBACK_BLOCKS]:
            parts.append(block.strip()[:DESCRIPTION_QUERY_CHARS])

    return " ".join(parts)


def retrieve(query: str, limit: int = DEFAULT_CANDIDATE_COUNT) -> list[str]:
    # Return the most likely taxonomy entries for a query, best first.
    categories, weights, idf = _index()

    # Term frequency is counted, not collapsed to a set. 
    query_counts = Counter(_tokenise(query))
    if not query_counts:
        return []

    query_tf = {
        token: 1.0 + math.log(count) for token, count in query_counts.items()
    }

    scored: list[tuple[float, int]] = []
    for index, token_weights in enumerate(weights):
        score = 0.0
        for token, frequency in query_tf.items():
            weight = token_weights.get(token)
            if weight:
                score += weight * idf.get(token, 0.0) * frequency
        if score > 0:
            scored.append((score / math.sqrt(len(token_weights) + 1), index))

    scored.sort(key=lambda pair: -pair[0])
    return [categories[index] for _score, index in scored[:limit]]


def is_valid(category: str) -> bool:
    """Is this string an exact taxonomy entry?"""
    categories, _weights, _idf = _index()
    return category in set(categories)
