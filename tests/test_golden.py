"""Golden assertions against the five provided PDPs.

These run against the cached snapshot in `out/products.json`, NOT against live model
calls, so the fast suite stays free and deterministic. Regenerate the snapshot with:

    uv run python main.py

Run these with `uv run pytest -m golden`.

Two rules govern what is asserted here, both of them about not fooling ourselves:

  1. PAGE-SPECIFIC EXPECTATIONS LIVE ONLY IN TESTS. Nothing in this file may leak into
     the extractor. The production code must stay generic, so the moment a value here
     would need a corresponding branch in `extract.py`, the design is wrong.

  2. ASSERT STABLE FACTS, NOT GENERATED PROSE. Name, brand, price, currency, category,
     identifiers, and variant relationships are properties of the page and will not
     drift. Descriptions and key features are model-authored and legitimately vary
     between runs; asserting their wording would produce a suite that fails for no
     reason and trains everyone to ignore it.
"""

import json
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).parent.parent / "out" / "products.json"

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def catalogue() -> dict[str, dict]:
    """The extraction snapshot, keyed by source filename."""
    if not SNAPSHOT.exists():
        pytest.skip(f"No snapshot at {SNAPSHOT}; run `uv run python main.py` first.")
    products = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {product["source"]: product for product in products}


def test_every_page_produced_a_product(catalogue):
    """All five pages must extract. A silent drop is the failure mode that matters."""
    expected = {"ace.html", "adaysmarch.html", "article.html", "llbean.html", "nike.html"}
    assert expected <= set(catalogue), f"missing: {expected - set(catalogue)}"


# --------------------------------------------------------------------------------
# Identity: name, brand, price, currency
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,brand,price,currency",
    [
        ("ace.html", "DeWalt", 129.00, "USD"),
        ("adaysmarch.html", "A Day's March", 170.00, "USD"),
        ("article.html", "Article", 349.00, "USD"),
        ("llbean.html", "L.L.Bean", 29.95, "USD"),
        ("nike.html", "Nike", 76.99, "GBP"),
    ],
)
def test_brand_and_price(catalogue, source, brand, price, currency):
    """Brand and price are facts about the page and must be exact.

    Note `nike.html` is priced in GBP: the currency must come from the page, not be
    assumed. A pipeline that defaulted everything to USD would pass four of these and
    be quietly wrong about the fifth.
    """
    product = catalogue[source]
    assert product["brand"] == brand
    assert product["price"]["price"] == pytest.approx(price)
    assert product["price"]["currency"] == currency


@pytest.mark.parametrize(
    "source,fragment",
    [
        ("ace.html", "DeWalt 20V MAX"),
        ("adaysmarch.html", "Miller Cotton Lyocell Trousers"),
        ("article.html", "Pilar Floor Lamp"),
        ("llbean.html", "Carefree Unshrinkable Tee"),
        ("nike.html", "Air Force 1"),
    ],
)
def test_product_name(catalogue, source, fragment):
    """The canonical product name must be recovered.

    A fragment rather than the full string, because merchants punctuate titles
    inconsistently and the meaningful assertion is that the right product was
    identified, not that a separator was reproduced.
    """
    assert fragment in catalogue[source]["name"]


def test_html_entities_are_decoded(catalogue):
    """No raw HTML entities may survive into product text.

    Several sites entity-encode the contents of their JSON-LD blocks even though the
    spec does not require it, which otherwise leaks a literal `&amp;` into names.
    """
    for source, product in catalogue.items():
        assert "&amp;" not in product["name"], source
        assert "&quot;" not in product["name"], source


# --------------------------------------------------------------------------------
# Category
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected,exact",
    [
        # Unambiguous: the taxonomy has one obviously correct leaf.
        ("ace.html", "Hardware > Tools > Drills > Handheld Power Drills", True),
        ("adaysmarch.html", "Apparel & Accessories > Clothing > Pants", True),
        ("nike.html", "Apparel & Accessories > Shoes", True),
        # Genuinely ambiguous, so only the branch is asserted. A floor lamp could
        # reasonably land on more than one lighting leaf, and a henley on more than
        # one top; pinning the exact leaf would test the model's taste, not the system.
        ("article.html", "Home & Garden > Lighting", False),
        ("llbean.html", "Apparel & Accessories > Clothing", False),
    ],
)
def test_category(catalogue, source, expected, exact):
    """Category must be correct, asserted exactly only where the answer is unambiguous."""
    actual = catalogue[source]["category"]["name"]
    if exact:
        assert actual == expected
    else:
        assert actual.startswith(expected), f"{actual!r} not under {expected!r}"


def test_all_categories_are_valid_taxonomy_entries(catalogue):
    """Every category must be a verbatim Google Product Taxonomy entry."""
    from models import VALID_CATEGORIES

    for source, product in catalogue.items():
        assert product["category"]["name"] in VALID_CATEGORIES, source


# --------------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------------


def test_nike_recovers_full_size_run(catalogue):
    """A ProductGroup with enumerated variants must be extracted exactly.

    This page declares its variants in schema.org with per-variant GTIN and price, so
    extraction needs no model at all and the result should be complete and precise.
    Note the page also lists eight further `hasVariant` entries that contain only a
    URL - those are other colourways, carry no describable configuration, and are
    correctly omitted rather than emitted as empty variants.
    """
    variants = catalogue["nike.html"]["variants"]
    assert len(variants) >= 15

    for variant in variants:
        assert "Size" in variant["options"]
        assert "Color" in variant["options"]
        assert variant["price"]["currency"] == "GBP"
        assert variant["gtin"], "schema.org supplied a GTIN per variant"

    # Sizes must be distinct: a duplicated size means the dedupe step failed.
    sizes = [variant["options"]["Size"] for variant in variants]
    assert len(sizes) == len(set(sizes))


def test_variants_are_never_cartesian_products(catalogue):
    """No product may list more configurations than its dimensions can produce.

    A cheap, universal upper bound on invented combinations: the number of variants
    can never exceed the product of the distinct values per dimension.
    """
    for source, product in catalogue.items():
        variants = product["variants"]
        if len(variants) < 2:
            continue

        dimensions: dict[str, set] = {}
        for variant in variants:
            for key, value in variant["options"].items():
                dimensions.setdefault(key, set()).add(value)

        maximum = 1
        for values in dimensions.values():
            maximum *= len(values)
        assert len(variants) <= maximum, f"{source}: implies invented combinations"


def test_every_variant_has_options(catalogue):
    """A variant with no options describes no configuration and must not exist."""
    for source, product in catalogue.items():
        for variant in product["variants"]:
            assert variant["options"], source


# --------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,minimum",
    [
        ("ace.html", 5),
        ("adaysmarch.html", 5),
        ("article.html", 8),
        ("llbean.html", 5),
        ("nike.html", 3),
    ],
)
def test_image_recall(catalogue, source, minimum):
    """Each page's gallery must be recovered, not just its OpenGraph hero image."""
    assert len(catalogue[source]["image_urls"]) >= minimum


def test_images_are_deduplicated_by_asset(catalogue):
    """The same photograph must never appear twice at two resolutions."""
    import images

    for source, product in catalogue.items():
        keys = [images._asset_key(url) for url in product["image_urls"]]
        assert len(keys) == len(set(keys)), f"{source}: duplicate assets emitted"


def test_article_images_are_full_resolution(catalogue):
    """Renditions must resolve to the largest the page offers, not the first seen.

    This page serves one photograph at a dozen widths from 320px upward. Emitting the
    320px thumbnail would satisfy a naive "did we get an image" check while failing
    the actual requirement, which is full resolution.
    """
    urls = catalogue["article.html"]["image_urls"]
    sized = [url for url in urls if "w=" in url]
    assert sized, "expected width-parameterised renditions on this page"
    for url in sized:
        width = int(url.split("w=")[1].split("&")[0])
        assert width >= 1000, f"low-resolution rendition emitted: {url}"


def test_no_cross_sell_images_in_primary_product(catalogue):
    """Imagery belonging to a different product must not be attributed to this one.

    This page embeds a cross-sell block whose `srcset` points at a different SKU. The
    identifier of the page's own product appears in its image paths, so any image
    carrying a *different* SKU identifier is contamination.
    """
    urls = catalogue["article.html"]["image_urls"]
    foreign = [url for url in urls if "SKU" in url and "SKU25289" not in url]
    assert not foreign, f"cross-sell imagery leaked in: {foreign}"


# --------------------------------------------------------------------------------
# Descriptive fields: presence only, never wording
# --------------------------------------------------------------------------------


def test_descriptive_fields_are_populated(catalogue):
    """Every product needs a usable description and some key features.

    Presence and plausibility only. The wording is model-authored and varies
    legitimately between runs, so asserting on it would make this suite flaky and
    teach everyone to ignore its failures.
    """
    for source, product in catalogue.items():
        assert len(product["description"]) > 40, source
        assert product["key_features"], source
        assert all(feature.strip() for feature in product["key_features"]), source
