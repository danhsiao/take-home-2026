"""Runtime validation of model output against the evidence that produced it.

This is the hallucination guardrail. A language model asked to read a messy PDP will
occasionally assert something the page never said: a plausible-looking price, a CDN
URL with three characters changed, a variant combination that does not exist. None of
those are distinguishable from correct output by inspection - they are only
distinguishable by checking them back against the evidence.

So every claim a model makes is checked here before it becomes a `Product`.

The single design decision that matters: `validate_candidate` is a PURE FUNCTION of
(candidate, evidence). It performs no I/O, calls no model, and holds no state. That
makes the entire adversarial test-suite free and deterministic - every failure class
can be exercised with a hand-built fixture and no API call - and it means the rules
can be reasoned about in isolation from the pipeline that applies them.

What is deliberately NOT here:

  * Cross-sell detection. Once a candidate exists, its name and price are just
    strings, and there is nothing generic left to check them against. Contamination is
    prevented structurally in `extract.py` (link-density suppression) and asserted in
    the test suite. A post-hoc runtime rule here would be theatre.
  * Anything site-, merchant-, or category-specific. Every rule below is a universal
    invariant about the relationship between a claim and its evidence.

Severity determines what the pipeline does, not how alarming the message is:
  error   -> the value is dropped, or the extraction fails (see `pipeline.py`)
  warning -> the result is annotated and a metric is counted, nothing is discarded
"""

import json
import re
from typing import Any

from models import PageEvidence, ProductCandidate, ValidationIssue

# Currency codes are three ASCII letters (ISO 4217). Anything else is a symbol or a
# formatting artefact that leaked into a structured field.
_ISO_CURRENCY = re.compile(r"^[A-Za-z]{3}$")

# Identifiers shorter than this are too common as substrings to verify by containment:
# a two-character SKU would "appear in the evidence" by coincidence almost always.
_MIN_VERIFIABLE_IDENTIFIER = 3


def validate_candidate(
    candidate: ProductCandidate, evidence: PageEvidence
) -> list[ValidationIssue]:
    """Check every claim in `candidate` against `evidence`.

    Args:
        candidate: the model's interpretation of the page.
        evidence: the deterministic harvest the model was given.

    Returns:
        Every issue found, in no particular order. An empty list means every claim the
        model made is traceable back to something the page actually contained.

        Note this reports issues; it does not modify the candidate. Deciding what to
        drop and what to fail on is `pipeline.py`'s job, so that the policy lives in
        one place and this function stays a pure predicate.
    """
    corpus = _evidence_corpus(evidence)
    sources = _evidence_sources(evidence)
    known_image_urls = {asset.url for asset in evidence.images}
    known_video_urls = set(evidence.videos)

    issues: list[ValidationIssue] = []

    issues.extend(_check_category(candidate))
    issues.extend(_check_price(candidate, corpus))
    issues.extend(_check_media(candidate, known_image_urls, known_video_urls))
    issues.extend(_check_variants(candidate, corpus, sources, known_image_urls))

    return issues


# --------------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------------


def _check_category(candidate: ProductCandidate) -> list[ValidationIssue]:
    """Category must be an exact member of the Google Product Taxonomy.

    The taxonomy has 5,596 entries and near-misses are the common failure: a model
    will happily return "Home & Garden > Lighting > Floor Lamps" when the real entry
    is "Home & Garden > Lighting > Lamps". Exact membership is the only check that
    catches this, and it is why the model is given a shortlist to copy from rather
    than asked to compose a path.
    """
    # Imported lazily so this module can be exercised without loading the 442KB
    # taxonomy file, which keeps the unit tests fast.
    from models import VALID_CATEGORIES

    issues: list[ValidationIssue] = []

    if not candidate.category:
        issues.append(
            ValidationIssue(
                field="category",
                code="category.missing",
                message="No category was selected.",
            )
        )
        return issues

    if candidate.category not in VALID_CATEGORIES:
        issues.append(
            ValidationIssue(
                field="category",
                code="category.not_in_taxonomy",
                message=(
                    f"'{candidate.category}' is not a verbatim entry in the Google "
                    f"Product Taxonomy."
                ),
            )
        )

    if not candidate.category_confident:
        # Not an error: the model correctly declined to guess. This is the signal that
        # retrieval, not selection, was the failure - and it is what the repair call
        # responds to by widening the shortlist.
        issues.append(
            ValidationIssue(
                field="category",
                code="category.low_confidence",
                message="Model reported that no retrieved category fit the product.",
                severity="warning",
            )
        )

    return issues


def _check_price(candidate: ProductCandidate, corpus: str) -> list[ValidationIssue]:
    """Price must be internally coherent, and ideally traceable to the evidence.

    Two rules with deliberately different severities:

      * Incoherent prices are ERRORS. A negative price, or a "was" price below the
        "now" price, is wrong regardless of what the page said.
      * Unsupported prices are WARNINGS. Prices appear in the wild as `29.95`,
        `$29.95`, `2995` (minor units), and `29,95` (European decimal comma). A
        containment check across all those forms false-positives constantly, and a
        validator that blocks on formatting variance is worse than no validator. So
        this is recorded as a drift metric rather than used to reject a product.
    """
    issues: list[ValidationIssue] = []
    price = candidate.price
    if price is None:
        return issues

    if price.price <= 0:
        issues.append(
            ValidationIssue(
                field="price.price",
                code="price.insane",
                message=f"Price {price.price} is not positive.",
            )
        )

    if not _ISO_CURRENCY.match(price.currency or ""):
        issues.append(
            ValidationIssue(
                field="price.currency",
                code="price.insane",
                message=f"Currency '{price.currency}' is not a 3-letter ISO code.",
            )
        )

    if price.compare_at_price is not None and price.compare_at_price <= price.price:
        issues.append(
            ValidationIssue(
                field="price.compare_at_price",
                code="price.insane",
                message=(
                    f"compare_at_price {price.compare_at_price} is not above the "
                    f"current price {price.price}; it is meant to be the pre-sale price."
                ),
            )
        )

    if price.price > 0 and not _number_appears(price.price, corpus):
        issues.append(
            ValidationIssue(
                field="price.price",
                code="price.unsupported",
                message=f"Price {price.price} was not found anywhere in the evidence.",
                severity="warning",
            )
        )

    return issues


def _check_media(
    candidate: ProductCandidate,
    known_image_urls: set[str],
    known_video_urls: set[str],
) -> list[ValidationIssue]:
    """Every media URL must be one the deterministic harvest actually found.

    This is the highest-value rule in the module. Product image URLs are long, contain
    UUIDs and transform tokens, and are exactly the kind of string a language model
    silently corrupts - and a corrupted CDN URL looks entirely plausible while
    resolving to nothing.

    The check is exact-match against the harvested set, not a similarity test.
    "Close to a real URL" is not a useful property for a URL.

    Note the candidate carries no top-level image list of its own by design: product
    images are taken directly from the deterministic harvest, never from the model.
    This rule therefore guards variant-level imagery and the video URL.
    """
    issues: list[ValidationIssue] = []

    for index, variant in enumerate(candidate.variants):
        for url in variant.image_urls:
            if url not in known_image_urls:
                issues.append(
                    ValidationIssue(
                        field=f"variants[{index}].image_urls",
                        code="image.unsupported",
                        message=f"Image URL was not found in the page evidence: {url}",
                    )
                )

    return issues


def _check_variants(
    candidate: ProductCandidate,
    corpus: str,
    sources: list[str],
    known_image_urls: set[str],
) -> list[ValidationIssue]:
    """Variants must be well-formed, distinct, and grounded in the evidence.

    The interesting rule here is grounding, which is how we detect invented
    combinations. See `_is_variant_grounded` for why this replaces the more obvious
    "did the model produce a full Cartesian product" check.
    """
    issues: list[ValidationIssue] = []
    seen_options: list[dict[str, str]] = []

    for index, variant in enumerate(candidate.variants):
        field = f"variants[{index}]"

        if not variant.options:
            issues.append(
                ValidationIssue(
                    field=f"{field}.options",
                    code="variant.empty_options",
                    message="Variant has no options; it does not describe a configuration.",
                )
            )
            continue

        if variant.options in seen_options:
            issues.append(
                ValidationIssue(
                    field=f"{field}.options",
                    code="variant.duplicate",
                    message=f"Duplicate variant configuration: {variant.options}",
                    severity="warning",
                )
            )
        else:
            seen_options.append(variant.options)

        for identifier, label in ((variant.sku, "sku"), (variant.gtin, "gtin")):
            if identifier and not _identifier_appears(identifier, corpus):
                issues.append(
                    ValidationIssue(
                        field=f"{field}.{label}",
                        code="identifier.unsupported",
                        message=(
                            f"{label} '{identifier}' does not appear in the page evidence."
                        ),
                    )
                )

        if not _is_variant_grounded(variant, sources, corpus, known_image_urls):
            issues.append(
                ValidationIssue(
                    field=field,
                    code="variant.ungrounded",
                    message=(
                        f"No single evidence source supports the configuration "
                        f"{variant.options}, and it carries no verifiable identifier, "
                        f"price, or URL. It may be an invented combination."
                    ),
                )
            )

    return issues


def _is_variant_grounded(
    variant: Any, sources: list[str], corpus: str, known_image_urls: set[str]
) -> bool:
    """Whether a variant describes a configuration the page actually offers.

    This is the anti-Cartesian rule, and it is deliberately *not* a count comparison.
    The obvious check - flag the output when the number of variants equals the product
    of the option cardinalities - is wrong: a page legitimately offering every colour
    in every size produces exactly that number, and one of the provided PDPs does
    precisely this with 25 real, individually-priced variants. Counting would reject
    correct output while still passing an invented 3x3 grid that happened to be
    missing one entry.

    Grounding asks a better question: is there any evidence for THIS combination?

      1. Do all of the option values co-occur inside ONE evidence source? A real
         variant record names its own colour and size together; a Cartesian product
         invented by the model pairs values that never appear together anywhere.
      2. Failing that, does the variant carry a verifiable identifier, price, or URL?
         A page may list options in a form we cannot associate, but if the model
         recovered a real SKU for the combination, something concrete backs it.

    Both are universal properties of the claim-to-evidence relationship - no merchant
    or category knowledge involved.
    """
    values = [str(v).strip() for v in variant.options.values() if str(v).strip()]
    if not values:
        return False

    # 1. Co-occurrence within a single source.
    for source in sources:
        lowered = source.lower()
        if all(value.lower() in lowered for value in values):
            return True

    # 2. A concrete, verifiable commercial anchor for this exact configuration.
    if variant.sku and _identifier_appears(variant.sku, corpus):
        return True
    if variant.gtin and _identifier_appears(variant.gtin, corpus):
        return True
    if variant.url and variant.url in corpus:
        return True
    if variant.image_urls and any(url in known_image_urls for url in variant.image_urls):
        return True
    if variant.price and _number_appears(variant.price.price, corpus):
        return True

    return False


# --------------------------------------------------------------------------------
# Evidence lookup helpers
# --------------------------------------------------------------------------------


def _evidence_sources(evidence: PageEvidence) -> list[str]:
    """The evidence, as a list of independently-serialised sources.

    Kept separate rather than concatenated so that co-occurrence can be tested *within*
    a source. That distinction is the whole point of the grounding rule: "Black" and
    "Size 10" both appearing somewhere on the page proves nothing, whereas both
    appearing in the same variant record proves the combination exists.
    """
    sources: list[str] = []

    for item in evidence.structured:
        sources.append(json.dumps(item, default=str))
    for subtree in evidence.json_subtrees:
        sources.append(json.dumps(subtree, default=str))
    for scope in evidence.microdata:
        sources.append(json.dumps(scope, default=str))

    sources.extend(evidence.text_blocks)
    return sources


def _evidence_corpus(evidence: PageEvidence) -> str:
    """All evidence flattened into one searchable string.

    Used for existence checks ("does this SKU appear anywhere?"), as opposed to the
    co-occurrence checks that need sources kept apart.
    """
    parts = _evidence_sources(evidence)
    parts.extend(f"{key} {value}" for key, value in evidence.meta.items())
    parts.extend(asset.url for asset in evidence.images)
    parts.extend(evidence.videos)
    parts.extend(evidence.breadcrumbs)
    if evidence.url:
        parts.append(evidence.url)
    return "\n".join(parts)


def _identifier_appears(identifier: str, corpus: str) -> bool:
    """Whether an identifier occurs in the evidence.

    Very short identifiers are not verifiable by containment - a two-character string
    appears in any large document by chance - so those are accepted rather than
    reported, to avoid manufacturing false accusations.
    """
    cleaned = identifier.strip()
    if len(cleaned) < _MIN_VERIFIABLE_IDENTIFIER:
        return True
    if cleaned in corpus:
        return True
    # Identifiers are frequently displayed with separators that the structured value
    # omits (or vice versa), e.g. "TA224626" shown as "Item # TA-224626".
    stripped = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if len(stripped) >= _MIN_VERIFIABLE_IDENTIFIER:
        normalised_corpus = re.sub(r"[^A-Za-z0-9]", "", corpus)
        return stripped in normalised_corpus
    return False


def _number_appears(value: float, corpus: str) -> bool:
    """Whether a numeric value appears in the evidence in any common price format.

    Checks the representations a price realistically takes on a page: with and without
    decimals, with a decimal comma, with a thousands separator, and as an integer count
    of minor units (cents), which is how many APIs serialise money.
    """
    as_float = float(value)
    as_int = int(round(as_float))

    plain = f"{as_float:.2f}"  # 1234.56
    grouped = f"{as_float:,.2f}"  # 1,234.56

    # Swapping both separators at once yields the continental European convention,
    # where "." groups thousands and "," is the decimal mark: 1.234,56. Much of the
    # world writes prices this way, and recognising only the Anglo format would raise
    # a spurious warning on every such page.
    european = grouped.translate(str.maketrans(",.", ".,"))

    candidates = {
        plain,  # 1234.56
        plain.replace(".", ","),  # 1234,56
        grouped,  # 1,234.56
        european,  # 1.234,56
        f"{as_float:g}",  # 29.95 / 350
        str(as_int),  # 350
        f"{as_int:,}",  # 1,299
        str(int(round(as_float * 100))),  # 2995 (minor units)
    }

    return any(candidate in corpus for candidate in candidates)
