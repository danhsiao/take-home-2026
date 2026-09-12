"""Checks what the model returned against the evidence that produced it.
"""

import json
import re
from typing import Any

from models import PageEvidence, ProductCandidate, ValidationIssue

# Currency codes are three ASCII letters (ISO 4217).
_ISO_CURRENCY = re.compile(r"^[A-Za-z]{3}$")

# Identifiers shorter than this are too common as substrings to verify by containment:
# a two-character SKU would "appear in the evidence" by coincidence almost always.
_MIN_VERIFIABLE_IDENTIFIER = 3


def validate_candidate(
    candidate: ProductCandidate, evidence: PageEvidence
) -> list[ValidationIssue]:
    # Check every claim in `candidate` against `evidence`.
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
    # The category has to be an exact entry in the Google Product Taxonomy.
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
        # Not an error: the model correctly declined to guess. 
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
    # The price has to make sense, and ideally appear somewhere in the evidence.

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
    # Every media URL has to be one the deterministic harvest actually found.
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
    # Variants have to be well formed, distinct, and backed by the evidence.  
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
    # Does this variant describe something the page really offers?

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
    # The evidence, as a list of separately serialised sources.
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
    # All the evidence flattened into one searchable string.
    parts = _evidence_sources(evidence)
    parts.extend(f"{key} {value}" for key, value in evidence.meta.items())
    parts.extend(asset.url for asset in evidence.images)
    parts.extend(evidence.videos)
    parts.extend(evidence.breadcrumbs)
    if evidence.url:
        parts.append(evidence.url)
    return "\n".join(parts)


def _identifier_appears(identifier: str, corpus: str) -> bool:
    # Does this identifier show up in the evidence?
    cleaned = identifier.strip()
    if len(cleaned) < _MIN_VERIFIABLE_IDENTIFIER:
        return True
    if cleaned in corpus:
        return True

    stripped = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if len(stripped) >= _MIN_VERIFIABLE_IDENTIFIER:
        normalised_corpus = re.sub(r"[^A-Za-z0-9]", "", corpus)
        return stripped in normalised_corpus
    return False


def _number_appears(value: float, corpus: str) -> bool:
    # Does this number show up in the evidence, in any common price format?
    as_float = float(value)
    as_int = int(round(as_float))

    plain = f"{as_float:.2f}"  # 1234.56
    grouped = f"{as_float:,.2f}"  # 1,234.56

    # Swapping both separators at once yields the continental European convention,
    # where "." groups thousands and "," is the decimal mark.
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
