"""The stateless worker primitive: raw HTML in, ExtractionResult out.

This module owns two things:

  1. FACTS DERIVATION - reading high-confidence values out of schema.org and
     OpenGraph, where the meaning of a field is fixed by a specification rather than
     guessed. This is the only place in the pipeline that assigns semantics
     deterministically, and it is legitimate precisely because the semantics were
     defined by someone else.

  2. ORCHESTRATION AND POLICY - running the stages in order, deciding what to do with
     each validation issue, and assembling the final `Product`.

The central rule of the merge is FACTS BEAT MODEL OUTPUT. If schema.org stated a
price, that is the price; the model's opinion is discarded (and the disagreement
logged, because a persistent gap between the two is a useful drift signal). A model
can fill gaps. It cannot overwrite the page's own machine-readable claims. That single
rule does more to prevent hallucinated output than any amount of prompt engineering.

Nothing here raises to the caller. Every failure - a transport error, an unfindable
price, an invalid category - becomes a structured `ExtractionResult` with the reason
attached, because a batch of 50 million products cannot afford to stop at the first
malformed page.
"""

import hashlib
import html as html_module
import logging
import re
from typing import Any

import extract
import normalize
import taxonomy
import validate
from models import (
    Category,
    DeterministicFacts,
    ExtractionResult,
    PageEvidence,
    Price,
    Product,
    ProductCandidate,
    ValidationIssue,
    Variant,
)

logger = logging.getLogger(__name__)

# schema.org availability values that mean "you can buy this right now".
_IN_STOCK = {"instock", "instoreonly", "onlineonly", "limitedavailability", "presale"}


async def run_html(html: str, url: str | None = None) -> ExtractionResult:
    """Extract one product from one page. The unit of work the whole system scales by.

    Deliberately stateless and side-effect free: no database, no queue, no shared
    cache. That is what lets the production design in the README wrap it in workers
    and horizontal scaling without touching this code.

    Args:
        html: raw page source.
        url: the page URL, if known.

    Returns:
        An ExtractionResult. `product` is None when a required field could not be
        resolved, with the reason in `errors`.
    """
    result = ExtractionResult(source=url)

    # --- Stage 1: deterministic harvest -------------------------------------------
    evidence = extract.run(html, url)
    facts = derive_facts(evidence)

    # --- Stage 2: deterministic taxonomy retrieval --------------------------------
    # `fallback_text` matters for generality: a page may carry no JSON-LD, no
    # breadcrumbs and no microdata at all, which is an ordinary input rather than a
    # broken one. Without a fallback such a page would be rejected here, before
    # inference had a chance to read anything - failing on exactly the sparse pages
    # the model is most needed for.
    query = taxonomy.build_query(
        name=facts.name,
        description=facts.description,
        breadcrumbs=evidence.breadcrumbs,
        declared_category=_declared_category(evidence),
        fallback_text=evidence.text_blocks,
    )
    candidates = taxonomy.retrieve(query)
    if not candidates:
        result.errors.append(
            "category.no_candidates: taxonomy retrieval found nothing for this page; "
            "it may not be a product detail page."
        )
        return result

    # --- Stage 3: one LLM call ----------------------------------------------------
    candidate = await normalize.run(evidence, facts, candidates)
    if candidate is None:
        result.errors.append("Normalization call failed; no candidate was produced.")
        return result

    # --- Stage 4: validate, repair, and enforce policy ----------------------------
    issues = validate.validate_candidate(candidate, evidence)

    if _needs_category_repair(issues):
        repaired = await normalize.repair_category(candidate, issues)
        if repaired:
            candidate.category = repaired
            candidate.category_confident = True
            # Re-validate rather than assume the repair worked: a second bad answer
            # must fail the product, not slip through unchecked.
            issues = validate.validate_candidate(candidate, evidence)
        else:
            # The repair declined, meaning no retrieved entry actually describes this
            # product. We stop here rather than keep the first guess. `Product`
            # requires a category and offers no "unknown" value, so a product whose
            # category cannot be determined genuinely cannot be represented - and a
            # category that is merely valid, rather than correct, is the worst
            # possible outcome, since nothing downstream can detect it.
            result.errors.append(
                f"category.unresolvable: no Google Product Taxonomy entry fits this "
                f"product (best rejected guess: {candidate.category!r})."
            )
            return result

    candidate = _apply_issue_policy(candidate, issues, result)

    # --- Stage 5: assemble --------------------------------------------------------
    result.product = _assemble(candidate, facts, evidence, result)
    return result


# --------------------------------------------------------------------------------
# Facts derivation
# --------------------------------------------------------------------------------


def derive_facts(evidence: PageEvidence) -> DeterministicFacts:
    """Read high-confidence values from standards-defined evidence.

    Only schema.org, microdata, and OpenGraph are consulted. Each of those defines
    what its fields mean, so reading them is interpretation-free. Visible text is
    deliberately not mined here - inferring "this string is the brand" from unlabelled
    markup is a guess, and guesses belong to the model, which at least reports its
    uncertainty.
    """
    facts = DeterministicFacts()

    products = extract.find_typed(evidence.structured, "Product", "ProductGroup")
    # Prefer the item with the most populated fields: pages sometimes carry a rich
    # Product alongside a stub for a related item.
    primary = max(products, key=lambda item: len(item), default=None)

    if primary:
        facts.name = _as_text(primary.get("name"))
        facts.brand = _brand_name(primary.get("brand"))
        facts.description = _as_text(primary.get("description"))
        facts.sku = _as_text(primary.get("sku") or primary.get("mpn"))
        facts.gtin = _first_gtin(primary)
        facts.price = _offer_price(primary.get("offers"))

    # OpenGraph fills gaps only; it never overrides schema.org, which is more specific.
    facts.name = facts.name or evidence.meta.get("og:title")
    facts.description = facts.description or evidence.meta.get("og:description")

    if facts.price is None:
        facts.price = _microdata_price(evidence)

    # Images and video are always deterministic. The model is never consulted about a
    # URL, so it can never corrupt one.
    facts.image_urls = [asset.url for asset in evidence.images]
    facts.video_url = evidence.videos[0] if evidence.videos else None

    facts.variants = _variants_from_structured(products)

    return facts


def _variants_from_structured(products: list[dict[str, Any]]) -> list[Variant]:
    """Build variants from a schema.org ProductGroup's `hasVariant` entries.

    This is the best case available: a `ProductGroup` declares `variesBy` and then
    enumerates real, individually-priced Products. When a page provides this, variant
    extraction needs no model at all, and the result is exact rather than inferred.

    `variesBy` names the dimensions as schema.org property URLs (".../color"), so the
    dimension labels come from the specification rather than from our own vocabulary.
    """
    variants: list[Variant] = []

    for group in products:
        raw_variants = group.get("hasVariant")
        if not isinstance(raw_variants, list):
            continue

        # Dimensions the group says it varies by, reduced to bare property names.
        varies_by = group.get("variesBy") or []
        if isinstance(varies_by, str):
            varies_by = [varies_by]
        dimensions = [str(v).rsplit("/", 1)[-1] for v in varies_by if v]
        # Fall back to the properties schema.org actually defines for variation.
        if not dimensions:
            dimensions = ["color", "size", "material", "pattern"]

        for entry in raw_variants:
            if not isinstance(entry, dict):
                continue

            options = {}
            for dimension in dimensions:
                value = _as_text(entry.get(dimension))
                if value:
                    # Title-case the dimension for a readable, presentable label.
                    options[dimension.replace("_", " ").title()] = value

            if not options:
                continue

            variants.append(
                Variant(
                    options=options,
                    sku=_as_text(entry.get("sku") or entry.get("mpn")),
                    gtin=_first_gtin(entry),
                    price=_offer_price(entry.get("offers")),
                    available=_availability(entry.get("offers")),
                    image_urls=[u for u in [_as_text(entry.get("image"))] if u],
                    url=_as_text(entry.get("url")) or _offer_url(entry.get("offers")),
                )
            )

    return variants


def _offer_price(offers: Any) -> Price | None:
    """Read a Price from a schema.org `offers` value.

    `offers` may be a single Offer, a list of Offers, or an AggregateOffer. When there
    are several, the lowest is taken, matching the "from $X" convention shoppers see.
    """
    if isinstance(offers, list):
        prices = [p for p in (_offer_price(o) for o in offers) if p]
        return min(prices, key=lambda p: p.price) if prices else None

    if not isinstance(offers, dict):
        return None

    # AggregateOffer nests the real offers one level down.
    if "offers" in offers and "price" not in offers:
        return _offer_price(offers["offers"])

    amount = _as_float(offers.get("price") or offers.get("lowPrice"))
    if amount is None or amount <= 0:
        return None

    currency = _as_text(offers.get("priceCurrency")) or ""

    # `compare_at_price` is deliberately NOT derived from `highPrice`. In an
    # AggregateOffer, highPrice is the top of a *range across variants*, not the
    # pre-sale price of this item - treating it as a former price invents a discount
    # wherever a product simply has a large size costing more. schema.org has no
    # standard former-price field, so that value is left to the model, which can read
    # an explicit "was" price off the page.
    return Price(
        price=amount,
        currency=currency.upper()[:3] if currency else "USD",
    )


def _offer_url(offers: Any) -> str | None:
    """Read a purchase URL from a schema.org `offers` value."""
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        return _as_text(offers.get("url"))
    return None


def _availability(offers: Any) -> bool | None:
    """Interpret schema.org availability as a simple boolean.

    Returns None rather than False when nothing is stated: "the page did not say" and
    "the page said out of stock" are different facts and must not be conflated.
    """
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if not isinstance(offers, dict):
        return None

    raw = _as_text(offers.get("availability"))
    if not raw:
        return None
    return raw.rsplit("/", 1)[-1].lower() in _IN_STOCK


def _microdata_price(evidence: PageEvidence) -> Price | None:
    """Recover a price from microdata, for pages using the older serialisation."""
    for scope in evidence.microdata:
        amount = _as_float(scope.get("price"))
        currency = _as_text(scope.get("priceCurrency"))
        if amount and amount > 0 and currency:
            return Price(price=amount, currency=currency.upper()[:3])
    return None


def _declared_category(evidence: PageEvidence) -> str | None:
    """The merchant's own category string, when schema.org carries one.

    A strong retrieval signal, and standards-defined rather than scraped.
    """
    for item in extract.find_typed(evidence.structured, "Product", "ProductGroup"):
        value = _as_text(item.get("category"))
        if value:
            return value
    return None


def _brand_name(brand: Any) -> str | None:
    """Read a brand name from schema.org, which allows a string or a Brand object."""
    if isinstance(brand, str):
        return brand.strip() or None
    if isinstance(brand, dict):
        return _as_text(brand.get("name"))
    if isinstance(brand, list) and brand:
        return _brand_name(brand[0])
    return None


def _first_gtin(item: dict[str, Any]) -> str | None:
    """Read any GTIN variant schema.org defines (gtin, gtin8/12/13/14)."""
    for key in ("gtin", "gtin14", "gtin13", "gtin12", "gtin8"):
        value = _as_text(item.get(key))
        if value:
            return value
    return None


def _as_text(value: Any) -> str | None:
    """Coerce a schema.org value to clean text, tolerating lists and nested objects.

    HTML entities are decoded because many sites entity-encode the contents of their
    JSON-LD blocks even though the spec does not call for it, which otherwise leaks
    literal `&amp;` into product names.
    """
    if isinstance(value, str):
        return html_module.unescape(value).strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and value:
        return _as_text(value[0])
    if isinstance(value, dict):
        return _as_text(value.get("name") or value.get("value") or value.get("@id"))
    return None


def _as_float(value: Any) -> float | None:
    """Coerce a schema.org price value to a float, tolerating formatted strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        # Keep only the numeric portion: values arrive as "$29.95" and "29.95 USD".
        match = re.search(r"\d+(?:\.\d+)?", cleaned)
        if match:
            return float(match.group(0))
    return None


# --------------------------------------------------------------------------------
# Validation policy and assembly
# --------------------------------------------------------------------------------


def _needs_category_repair(issues: list[ValidationIssue]) -> bool:
    """Whether the category warrants a second, cheap call."""
    return any(issue.code.startswith("category.") for issue in issues)


def _apply_issue_policy(
    candidate: ProductCandidate, issues: list[ValidationIssue], result: ExtractionResult
) -> ProductCandidate:
    """Act on validation findings: drop what is unsupported, record what is suspect.

    The policy is deliberately asymmetric, because the right response depends on the
    failure:

      * An unsupported image URL or identifier is DROPPED. Re-asking the model would
        resample the same distribution that produced the error; removing the
        unverifiable value is both cheaper and strictly more correct.
      * An ungrounded or malformed variant is DROPPED, with a warning. Better to
        under-report variants than to publish combinations the page never offered.
      * A duplicate variant is COLLAPSED.
      * Warnings are recorded and counted, never acted on.
    """
    dropped_variants: set[int] = set()
    dropped_images: set[tuple[int, str]] = set()

    for issue in issues:
        if issue.severity == "warning":
            result.warnings.append(f"{issue.code}: {issue.message}")
            continue

        index = _variant_index(issue.field)
        if issue.code in ("variant.ungrounded", "variant.empty_options") and index is not None:
            dropped_variants.add(index)
        elif issue.code == "identifier.unsupported" and index is not None:
            # Clear just the offending identifier; the configuration itself may be real.
            field = issue.field.rsplit(".", 1)[-1]
            if index < len(candidate.variants):
                setattr(candidate.variants[index], field, None)
            result.warnings.append(f"{issue.code}: {issue.message}")
        elif issue.code == "image.unsupported" and index is not None:
            dropped_images.add((index, issue.message.rsplit(": ", 1)[-1]))
            result.warnings.append(f"{issue.code}: {issue.message}")
        elif issue.code == "price.insane":
            # Discard the incoherent price rather than record a fatal error. A
            # structured price usually supersedes it anyway, and if none exists the
            # product will fail cleanly as `field.unresolvable` during assembly -
            # which is the honest outcome, and avoids reporting an error on a page
            # that in fact extracted correctly.
            candidate.price = None
            result.warnings.append(f"{issue.code}: {issue.message}")
        else:
            result.errors.append(f"{issue.code}: {issue.message}")

    for index, url in dropped_images:
        if index < len(candidate.variants):
            variant = candidate.variants[index]
            variant.image_urls = [u for u in variant.image_urls if u != url]

    if dropped_variants:
        result.warnings.append(
            f"Dropped {len(dropped_variants)} variant(s) unsupported by page evidence."
        )
        candidate.variants = [
            variant
            for index, variant in enumerate(candidate.variants)
            if index not in dropped_variants
        ]

    # Collapse duplicate configurations, keeping the first occurrence.
    seen: list[dict[str, str]] = []
    unique: list[Variant] = []
    for variant in candidate.variants:
        if variant.options not in seen:
            seen.append(variant.options)
            unique.append(variant)
    candidate.variants = unique

    return candidate


def _variant_index(field: str) -> int | None:
    """Recover the variant index from a validation issue's field path."""
    if not field.startswith("variants["):
        return None
    try:
        return int(field.split("[", 1)[1].split("]", 1)[0])
    except (IndexError, ValueError):
        return None


def _assemble(
    candidate: ProductCandidate,
    facts: DeterministicFacts,
    evidence: PageEvidence,
    result: ExtractionResult,
) -> Product | None:
    """Merge deterministic facts with model output into the final Product.

    Precedence is the whole point: for every field a standard could state, the
    deterministic value wins and the model's is used only to fill a gap. Where the two
    disagree on price, the disagreement is recorded - a persistent gap between what
    schema.org says and what a model reads off the page is exactly the kind of drift
    worth alerting on in production.

    Returns None when a required field cannot be resolved. That is a real outcome, not
    a defect: one of the provided pages states its price only in rendered markup, and
    the honest response to an unfindable required value is to fail loudly rather than
    let a model supply a plausible number.
    """
    name = facts.name or candidate.name
    brand = facts.brand or candidate.brand
    description = candidate.description or facts.description
    price = facts.price or candidate.price

    if facts.price and candidate.price and abs(facts.price.price - candidate.price.price) > 0.01:
        result.warnings.append(
            f"price.disagreement: structured data says {facts.price.price}, "
            f"model read {candidate.price.price}; using structured data."
        )

    # Variants from schema.org are exact; the model's are inferred. Prefer the former
    # when a page provided them, and fall back to interpretation when it did not.
    variants = facts.variants or candidate.variants

    # Colours are derived from variants where possible rather than asked for twice.
    colors = _colors_from(variants) or candidate.colors

    missing = [
        label
        for label, value in (
            ("name", name),
            ("brand", brand),
            ("description", description),
            ("price", price),
            ("category", candidate.category),
        )
        if not value
    ]
    if missing:
        result.errors.append(
            f"field.unresolvable: required field(s) not found on the page: "
            f"{', '.join(missing)}"
        )
        return None

    try:
        return Product(
            name=name,
            price=price,
            description=description,
            key_features=candidate.key_features,
            image_urls=facts.image_urls,
            video_url=facts.video_url,
            category=Category(name=candidate.category),
            brand=brand,
            colors=colors,
            variants=variants,
        )
    except Exception as error:  # noqa: BLE001 - a schema failure is a data problem
        result.errors.append(f"schema.invalid: {error}")
        return None


def _colors_from(variants: list[Variant]) -> list[str]:
    """Collect the distinct colour values named across variants.

    Derived rather than requested separately, so that product-level colours cannot
    contradict the variants they summarise.
    """
    colors: list[str] = []
    for variant in variants:
        for key, value in variant.options.items():
            if key.strip().lower() in ("color", "colour") and value not in colors:
                colors.append(value)
    return colors


def product_id(url: str | None, name: str | None) -> str:
    """A stable identifier for a product, used by the API and as a cache key.

    Content-hashed rather than sequential so it is reproducible across runs and across
    machines - the same page always yields the same id, which is what makes it usable
    as the fingerprint for the caching strategy described in the README.
    """
    basis = (url or name or "").strip().lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
