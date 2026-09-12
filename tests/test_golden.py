"""Golden assertions against the five provided PDPs.

These run against the cached snapshot in `out/products.json`, NOT against live model
calls, so the fast suite stays free and deterministic. Regenerate the snapshot with:

    uv run python main.py

Run these with `uv run pytest -m golden`.
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


@pytest.mark.parametrize(
    "source,maximum",
    [
        ("ace.html", 8),
        ("adaysmarch.html", 8),
        ("article.html", 14),
        ("llbean.html", 15),
        ("nike.html", 10),
    ],
)
def test_image_precision(catalogue, source, maximum):
    """Recall has an upper bound too.

    A gallery several times the size of the real one means the harvest is picking up
    swatches, marketing assets or a neighbouring product - which reads as a bug on the
    product page even though every individual URL resolves.
    """
    assert len(catalogue[source]["image_urls"]) <= maximum


def test_no_interface_assets_in_galleries(catalogue):
    """Icons, chevrons and badges ship as SVG; photographs do not."""
    for source, product in catalogue.items():
        svg = [url for url in product["image_urls"] if ".svg" in url.lower()]
        assert not svg, f"{source}: interface assets in gallery: {svg}"


def test_thumbnail_renditions_only_appear_when_nothing_larger_exists(catalogue):
    """A small rendition is a gallery image only when the page exposes no larger one.

    The rule used to be absolute: no rendition below `MIN_PRODUCT_WIDTH`, ever. That
    holds on a server-rendered page, where the markup carries the real gallery. It
    fails on a client-rendered one, where the markup carries a thumbnail and
    JavaScript swaps in the full-size image at runtime - the largest URL the saved
    HTML contains *is* a thumbnail. Enforcing the minimum there deletes the merchant's
    entire gallery for being small rather than for being wrong, leaving only the
    nominated hero.

    So the minimum still runs first and unmodified, and small renditions are admitted
    only by the fallback in `images.collect`: the strict pass left no gallery at all,
    while the product's own region visibly renders photographs. This test asserts that
    narrower property - a page that keeps a full-size gallery must not also carry
    thumbnails of it.
    """
    import images

    for source, product in catalogue.items():
        urls = product["image_urls"]
        widths = [images._effective_width(url) for url in urls]
        small = [
            url
            for url, width in zip(urls, widths)
            if width is not None and width < images.MIN_PRODUCT_WIDTH
        ]
        if not small:
            continue
        # Admitted only as a whole-gallery fallback, never mixed in beside full-size
        # photography of the same product.
        full_size = [
            url
            for url, width in zip(urls, widths)
            if width is not None and width >= images.MIN_PRODUCT_WIDTH
        ]
        assert len(full_size) <= 1, (
            f"{source}: thumbnails emitted alongside a full-size gallery: {small}"
        )


def test_galleries_stay_within_the_merchant_image_host(catalogue):
    """Every image is served by the same host the merchant nominated its hero from.

    Marketing banners, review photos and analytics beacons live elsewhere, and
    "elsewhere" is what separates them from product photography.

    This asserts the *host*, not the asset root. It previously asserted a single asset
    root, which turned out to encode an assumption that is simply not true of the web:
    a merchant may publish its nominated sharing image through one route
    (`/seo/<slug>.jpg`) and serve the gallery from a delivery route
    (`/asr/<uuid>.jpg`). Both are that merchant's product photography. One of the
    pages in `data/` does exactly this, and under the old assertion the only way to
    pass was to discard its entire real gallery and keep the cross-sell thumbnails
    that happened to share the sharing route - which is what the extractor used to do.

    The host is the part that genuinely distinguishes the merchant's own imagery from
    a third party's, so that is what is asserted now.
    """
    from urllib.parse import urlparse

    for source, product in catalogue.items():
        hosts = {urlparse(url).netloc.lower() for url in product["image_urls"]}
        assert len(hosts) == 1, f"{source}: images span several hosts: {hosts}"


def test_gallery_presents_one_configuration(catalogue):
    """This page embeds a second item code's full-size photography.

    Its Tall cut is filed under its own item number, and its photographs are the same
    shots re-registered under that number. Both sets are full resolution on the same
    host in the same folder, so only the id separates them - which makes this the
    regression test for the id gate. The Tall combinations keep their own imagery on
    their variant rows.
    """
    urls = catalogue["llbean.html"]["image_urls"]
    foreign = [url for url in urls if "224625" in url]
    assert not foreign, f"sibling product imagery leaked in: {foreign}"


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
