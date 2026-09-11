"""Google Product Taxonomy retrieval.

`Product.category` must be a verbatim entry in a 5,596-line taxonomy. Two approaches
are obviously wrong:

  * Let the model compose a category path. Near-misses are the normal outcome
    ("... > Floor Lamps" when the real entry is "... > Lamps"), and near-misses fail
    the `Category` validator, so most products would need a repair round-trip.
  * Send the whole taxonomy in the prompt. It is roughly 120,000 tokens. At 50M
    products that is several hundred thousand dollars of input tokens to answer one
    multiple-choice question per page.

So classification is split: this module deterministically narrows 5,596 entries to a
few dozen candidates, and the model only has to *choose* from that shortlist. The
retrieval is lexical, which matters for three reasons - it needs no embedding
infrastructure, it is perfectly reproducible, and it can be unit-tested directly
(recall@K is measurable without calling a model at all).

The model is also given an explicit "none of these fit" escape. That is what
distinguishes a bad *retrieval* from a bad *choice*: without it, the two failures are
indistinguishable in the output and neither can be diagnosed.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

CATEGORIES_FILE = Path(__file__).parent / "categories.txt"

DEFAULT_CANDIDATE_COUNT = 30
# Used by the repair call, alongside model-supplied keywords. Widening alone is not
# enough when the correct entry is absent for vocabulary reasons rather than ranking
# ones, which is why the repair path re-queries rather than simply asking again.
WIDE_CANDIDATE_COUNT = 100

# How much description text feeds the retrieval query. Enough to identify the product
# type, short enough that specification details do not drown it out.
DESCRIPTION_QUERY_CHARS = 120

# How many visible-text blocks to fall back on when a page declares nothing
# structured. Text blocks arrive ranked by proximity to the title, so the first few
# are the ones most likely to describe the product itself.
FALLBACK_BLOCKS = 5

# Tokens that carry no discriminative signal in a taxonomy of product categories.
# Deliberately tiny - aggressive stopword lists remove real signal ("bags", "parts").
_STOPWORDS = {"and", "or", "the", "a", "an", "of", "for", "with", "in", "other", "&"}

# Leaf segment matches count for more than ancestor matches: a query word matching
# "Drills" is far stronger evidence than one matching "Hardware".
_LEAF_WEIGHT = 2.5


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens, dropping stopwords.

    Singular/plural is normalised crudely by trimming a trailing "s", so that a query
    for "trousers" matches a category named "Trouser" and vice versa. This is cheap and
    language-specific, but the taxonomy itself is English, so it costs nothing to
    assume that here.
    """
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        tokens.append(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)
    return tokens


@lru_cache(maxsize=1)
def _index() -> tuple[list[str], list[dict[str, float]], dict[str, float]]:
    """Build the retrieval index once, and cache it for the process lifetime.

    Returns:
        categories: every taxonomy entry, in file order.
        weights:    per-category token -> weight, leaf tokens weighted higher.
        idf:        token -> inverse document frequency across the taxonomy.

    Caching matters at scale: the index is built once per worker and then reused for
    every product that worker handles, so the 442KB file is parsed once, not 50M times.
    """
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
    """Assemble the text used to retrieve taxonomy candidates.

    Breadcrumbs and any merchant-declared category are repeated, which weights them
    more heavily in the token counts. That is intentional: a merchant's own
    categorisation of its own product is by far the strongest signal available, and
    both are standards-defined (schema.org `BreadcrumbList` and `Product.category`)
    rather than scraped from styling.

    The brand is deliberately excluded from the retrieval text even though it is
    accepted as an argument, because a brand name matches taxonomy entries for reasons
    unrelated to what the product is - brands are routinely ordinary nouns.

    `fallback_text` is used only when the structured signals are absent entirely. A
    page may carry no JSON-LD, no breadcrumbs, and no microdata - that is a normal
    input, not a broken one - and such a page must still reach inference rather than
    being rejected for lack of a retrieval query. Whatever the page displays is a
    weaker signal than a declared category, but it is far better than nothing.
    """
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
        # A hard character cap rather than "the first sentence". Product descriptions
        # are frequently bulleted specification lists with no sentence boundary at all,
        # so sentence splitting can swallow the entire blob - and then construction
        # vocabulary ("button", "topstitch", "pockets") outweighs the product type and
        # drags retrieval toward entirely the wrong branch of the taxonomy.
        parts.append(description.strip()[:DESCRIPTION_QUERY_CHARS])

    # Only reached when nothing structured was available. Kept strictly as a last
    # resort so it can never dilute a page that did declare its own categorisation.
    if not parts and fallback_text:
        for block in fallback_text[:FALLBACK_BLOCKS]:
            parts.append(block.strip()[:DESCRIPTION_QUERY_CHARS])

    return " ".join(parts)


def retrieve(query: str, limit: int = DEFAULT_CANDIDATE_COUNT) -> list[str]:
    """Return the most plausible taxonomy entries for a query, best first.

    Scoring is IDF-weighted token overlap, normalised by category length so that short,
    specific entries are not beaten by long ones that merely contain more words.

    Args:
        query: free text, from `build_query`.
        limit: how many candidates to return.

    Returns:
        Up to `limit` verbatim taxonomy entries. Empty only if the query has no
        usable tokens at all.
    """
    categories, weights, idf = _index()

    # Term frequency is counted, not collapsed to a set. `build_query` deliberately
    # repeats the strongest signals (breadcrumbs, a merchant-declared category), and
    # de-duplicating the tokens here would silently discard that weighting - the
    # repetition would cost nothing and buy nothing.
    #
    # Sublinear (1 + log tf) damping rather than raw counts, so a term repeated three
    # times counts for more than one repeated twice, without a long name dominating
    # purely by restating the same word.
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
            # Normalising by category size stops "Apparel & Accessories > Clothing >
            # Outerwear > Coats & Jackets" from outranking "Coats & Jackets" purely for
            # having more tokens to match against.
            scored.append((score / math.sqrt(len(token_weights) + 1), index))

    scored.sort(key=lambda pair: -pair[0])
    return [categories[index] for _score, index in scored[:limit]]


def is_valid(category: str) -> bool:
    """Whether a string is a verbatim taxonomy entry."""
    categories, _weights, _idf = _index()
    return category in set(categories)
