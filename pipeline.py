"""The worker: HTML in, ExtractionResult out.

It does two jobs. It reads the values a published standard already stated, and it
runs the stages, applies the merge rules and builds the final Product.
"""

import hashlib
import html as html_module
import logging
import re
from typing import Any

import extract
import images
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
    # Extract one product from one page. This is the unit the system scales by.
    result = ExtractionResult(source=url)

    # --- Stage 1: deterministic harvest -------------------------------------------
    evidence = extract.run(html, url)
    facts = derive_facts(evidence)

    # --- Stage 2: deterministic taxonomy retrieval --------------------------------
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
            # category we cannot work out genuinely cannot be represented. And a
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
    # Read the high confidence values out of standards based evidence.

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

    facts.variants = (
        _variants_from_structured(products)
        or _variants_from_graph(evidence, facts.price)
        or _variants_from_controls(evidence)
    )

    return facts


def _variants_from_structured(products: list[dict[str, Any]]) -> list[Variant]:
    # Build variants from a schema.org ProductGroup's `hasVariant` list.

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


def _variants_from_graph(evidence: PageEvidence, price: Price | None) -> list[Variant]:
    # Turn the page's joined variant graph into Variants.

    currency = price.currency if price else None

    return [
        Variant(
            options=record.options,
            sku=record.sku,
            price=(
                Price(price=record.amount, currency=currency)
                if record.amount is not None and currency
                else None
            ),
            available=record.available,
            image_urls=images.align_to_assets(record.image_urls, evidence.images),
            url=record.url,
        )
        for record in evidence.variant_graph
    ]


def _variants_from_controls(evidence: PageEvidence) -> list[Variant]:
    # Build variants from the option groups the rendered page shows.

    # A page's controls are not all product options. 
    groups = [
        group
        for group in evidence.option_groups
        if len(group.values) > 1
        and any(
            value.image_url or value.url or value.available is not None
            for value in group.values
        )
    ]
    if not groups:
        return []
    inside = [group for group in groups if group.title_depth is not None]
    groups = inside or groups

    primary = max(groups, key=lambda group: len(group.values))

    pinned: dict[str, str] = {}
    for group in evidence.option_groups:
        if group is primary or not group.name:
            continue
        chosen = next((value for value in group.values if value.selected), None)
        if chosen:
            pinned[group.name] = chosen.label

    dimension = primary.name or "Option"
    variants: list[Variant] = []
    for value in primary.values:
        options = dict(pinned)
        options[dimension] = value.label
        variants.append(
            Variant(
                options=options,
                available=value.available,
                # The swatch is this configuration's own photograph. It is a small
                # rendition, so it is restated at the best resolution the page exposed
                # for that same asset, exactly as the graph path does.
                image_urls=images.align_to_assets(
                    [value.image_url] if value.image_url else [], evidence.images
                ),
                url=value.url,
            )
        )
    return variants


def _offer_price(offers: Any) -> Price | None:
    # Read a Price out of a schema.org `offers` value.

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
    # old price of this item. Treating it as one invents a discount
    # wherever a product simply has a large size costing more. schema.org has no
    # standard former-price field, so that value is left to the model, which can read
    # an explicit "was" price off the page.
    return Price(
        price=amount,
        currency=currency.upper()[:3] if currency else "USD",
    )


def _offer_url(offers: Any) -> str | None:
    # Read a buy URL out of a schema.org `offers` value.
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        return _as_text(offers.get("url"))
    return None


def _availability(offers: Any) -> bool | None:
    # Turn schema.org availability into a plain boolean.
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if not isinstance(offers, dict):
        return None

    raw = _as_text(offers.get("availability"))
    if not raw:
        return None
    return raw.rsplit("/", 1)[-1].lower() in _IN_STOCK


def _microdata_price(evidence: PageEvidence) -> Price | None:
    # Recover a price from microdata, for pages using the older format.
    for scope in evidence.microdata:
        amount = _as_float(scope.get("price"))
        currency = _as_text(scope.get("priceCurrency"))
        if amount and amount > 0 and currency:
            return Price(price=amount, currency=currency.upper()[:3])
    return None


def _declared_category(evidence: PageEvidence) -> str | None:
    # The shop's own category string, when schema.org carries one.
    for item in extract.find_typed(evidence.structured, "Product", "ProductGroup"):
        value = _as_text(item.get("category"))
        if value:
            return value
    return None


def _brand_name(brand: Any) -> str | None:
    # Read a brand name, which schema.org allows as a string or an object.
    if isinstance(brand, str):
        return brand.strip() or None
    if isinstance(brand, dict):
        return _as_text(brand.get("name"))
    if isinstance(brand, list) and brand:
        return _brand_name(brand[0])
    return None


def _first_gtin(item: dict[str, Any]) -> str | None:
    # Read any of the GTIN fields schema.org defines.
    for key in ("gtin", "gtin14", "gtin13", "gtin12", "gtin8"):
        value = _as_text(item.get(key))
        if value:
            return value
    return None


def _as_text(value: Any) -> str | None:
    # Turn a schema.org value into clean text, coping with lists and objects.
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
    # Turn a schema.org price into a float, coping with formatted strings.
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
    """Is the category bad enough to be worth a second, cheap call?"""
    return any(issue.code.startswith("category.") for issue in issues)


def _apply_issue_policy(
    candidate: ProductCandidate, issues: list[ValidationIssue], result: ExtractionResult
) -> ProductCandidate:
    # Act on the validator's findings. Drop what is unsupported, log what is odd.
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
            # Discard the incoherent price rather than record a fatal error. 
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
    # Pull the variant index back out of an issue's field path.
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
    #Merge the deterministic facts with the model output into the final Product.
    name = facts.name or candidate.name
    brand = facts.brand or candidate.brand
    description = candidate.description or facts.description
    price = facts.price or candidate.price

    if facts.price and candidate.price and abs(facts.price.price - candidate.price.price) > 0.01:
        result.warnings.append(
            f"price.disagreement: structured data says {facts.price.price}, "
            f"model read {candidate.price.price}; using structured data."
        )

    variants = facts.variants or candidate.variants

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
    # Collect the distinct colours named across the variants.
    colors: list[str] = []
    for variant in variants:
        for key, value in variant.options.items():
            if _COLOR_DIMENSION.search(key) and value not in colors:
                colors.append(value)
    return colors


_COLOR_DIMENSION = re.compile(r"\bcolou?rs?\b", re.I)


def product_id(url: str | None, name: str | None) -> str:
    #A stable id for a product, used by the API and as a cache key.

    basis = (url or name or "").strip().lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
