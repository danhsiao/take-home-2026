"""Falsification tests: PDP shapes deliberately unlike the supplied pages.

The supplied pages are five samples from a space of millions. A suite built only from
them would measure memorisation, so this module does the opposite: it invents page
shapes, retailers, and product categories that appear nowhere in `data/`, and asserts
that the same rules still hold.

The standard every rule in the extractor has to meet is: if the five supplied files
were deleted tomorrow and replaced with five different retailers in different
categories, would this rule still make conceptual sense? Each test below is one attempt
to answer no.

Categories used here (groceries, books, bicycles, cosmetics, industrial supply) are
chosen precisely because none of them appears in the provided data. No API calls.
"""

import pytest

import extract
import taxonomy
from models import PageEvidence, Price, ProductCandidate, Variant
from validate import validate_candidate


# --------------------------------------------------------------------------------
# Serialisation shapes we were not given
# --------------------------------------------------------------------------------


def test_microdata_only_page():
    """A page with microdata and no JSON-LD at all must still yield price and identity.

    Microdata is the older schema.org serialisation. None of the supplied pages use it
    as their primary carrier, so nothing in the extractor may quietly assume JSON-LD.
    """
    html = """
    <html><head><link rel="canonical" href="https://grocer.test/p/olive-oil"></head>
    <body><div itemscope itemtype="https://schema.org/Product">
      <h1 itemprop="name">Cold Pressed Olive Oil 500ml</h1>
      <span itemprop="brand">Grove Press</span>
      <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
        <meta itemprop="price" content="14.50">
        <meta itemprop="priceCurrency" content="EUR">
      </div>
      <span itemprop="sku">OO-500</span>
    </div></body></html>
    """
    evidence = extract.run(html)
    assert evidence.microdata, "microdata itemscopes must be harvested"

    import pipeline

    facts = pipeline.derive_facts(evidence)
    assert facts.price is not None
    assert facts.price.price == pytest.approx(14.50)
    assert facts.price.currency == "EUR"


def test_completely_unstructured_page_still_reaches_inference():
    """A page with no structured data whatsoever must not be rejected before inference.

    This is the sparse-PDP case: no JSON-LD, no microdata, no OpenGraph, no
    breadcrumbs, no framework state. Plenty of small retailers ship exactly this. The
    deterministic layer will find almost nothing, which is fine - but the page must
    still produce a usable taxonomy query, because rejecting it here would fail
    precisely the pages where the model is most needed.
    """
    html = """
    <html><head><title>Touring Bicycle Frame 56cm</title></head><body>
      <h1>Touring Bicycle Frame 56cm</h1>
      <p>Steel touring frame with rack mounts and a lifetime warranty.</p>
      <div>$899.00</div>
    </body></html>
    """
    evidence = extract.run(html)
    assert evidence.text_blocks, "visible text is the only evidence such a page has"

    query = taxonomy.build_query(
        name=None,
        description=None,
        breadcrumbs=evidence.breadcrumbs,
        declared_category=None,
        fallback_text=evidence.text_blocks,
    )
    assert query.strip(), "fallback must produce a non-empty query"
    assert taxonomy.retrieve(query, 30), "a sparse page must still retrieve candidates"


def test_variant_table_under_generic_key():
    """A homogeneous option table must be found by shape, under any key name.

    The key here is `variants`, the records describe a cosmetic product by shade, and
    the site is fictional. Nothing about recognising a record table should depend on
    which of those is true.
    """
    html = """
    <script type="application/json">
    {"product":{"title":"Tinted Lip Balm","prices":[{"amount":12.0}],
     "variants":[{"sku":"LB-01","shade":"Rose","price":12.0},
                 {"sku":"LB-02","shade":"Coral","price":12.0},
                 {"sku":"LB-03","shade":"Plum","price":12.0}]}}
    </script>
    """
    subtrees = extract.run(html).json_subtrees
    blob = str(subtrees)
    assert subtrees, "a product record with an option table must be discovered"
    for sku in ("LB-01", "LB-02", "LB-03"):
        assert sku in blob


def test_qualified_and_plural_field_names_are_recognised():
    """Records naming their fields `currentPrice`/`displayName` must still be found.

    Framework field names are routinely qualified by state or pluralised. A matcher
    keyed to the bare words `price` and `name` would be blind to a large fraction of
    real pages for no principled reason.
    """
    html = """
    <script>window.__STATE__ = {"catalog":{"item":
      {"displayName":"Hardcover Notebook A5","currentPrice":18.99,"itemId":"NB-A5"}}};
    </script>
    """
    subtrees = extract.run(html).json_subtrees
    assert any("NB-A5" in str(s) for s in subtrees), subtrees


def test_european_decimal_comma_price_is_supported_evidence():
    """A price written `1.234,56` must count as support for the value 1234.56.

    Decimal commas and dot thousands separators are the norm across much of the world.
    A validator that only recognised the Anglo format would emit spurious warnings on
    every such page.
    """
    evidence = PageEvidence(text_blocks=["Preis: 1.234,56 EUR"])
    candidate = ProductCandidate(
        name="Industrial Air Compressor",
        price=Price(price=1234.56, currency="EUR"),
        category="Business & Industrial",
    )
    codes = {issue.code for issue in validate_candidate(candidate, evidence)}
    assert "price.unsupported" not in codes


# --------------------------------------------------------------------------------
# Grounding rules must not depend on the vocabulary of any one category
# --------------------------------------------------------------------------------


def test_grounding_works_for_non_apparel_dimensions():
    """Variant dimensions unlike colour and size must ground identically.

    Nothing in the grounding rule may assume apparel vocabulary. Here the dimensions
    are voltage and kit contents, and the rule must behave exactly as it does for a
    shirt: real combinations pass, invented pairings fail.
    """
    evidence = PageEvidence(
        json_subtrees=[
            {"name": "Impact Driver", "sku": "ID-18-BARE", "voltage": "18V", "kit": "Bare Tool"},
            {"name": "Impact Driver", "sku": "ID-36-FULL", "voltage": "36V", "kit": "Full Kit"},
        ]
    )
    candidate = ProductCandidate(
        name="Impact Driver",
        category="Hardware > Tools",
        variants=[
            Variant(options={"Voltage": "18V", "Kit": "Bare Tool"}, sku="ID-18-BARE"),
            Variant(options={"Voltage": "36V", "Kit": "Full Kit"}, sku="ID-36-FULL"),
            # Never offered: the Cartesian filler between the two real records.
            Variant(options={"Voltage": "18V", "Kit": "Full Kit"}),
        ],
    )
    issues = validate_candidate(candidate, evidence)
    ungrounded = [issue for issue in issues if issue.code == "variant.ungrounded"]
    assert len(ungrounded) == 1
    assert ungrounded[0].field == "variants[2]"


def test_single_dimension_variants_ground_correctly():
    """A product varying in one dimension only must not be treated as suspicious.

    Books, food, and many hardware items vary along a single axis. The grounding rule
    must handle a one-dimensional option space as naturally as a two-dimensional one.
    """
    evidence = PageEvidence(
        text_blocks=["Available formats: Hardcover, Paperback, Audiobook"]
    )
    candidate = ProductCandidate(
        name="A Novel",
        category="Media > Books",
        variants=[
            Variant(options={"Format": "Hardcover"}),
            Variant(options={"Format": "Paperback"}),
            Variant(options={"Format": "Audiobook"}),
        ],
    )
    assert not [i for i in validate_candidate(candidate, evidence) if i.severity == "error"]


# --------------------------------------------------------------------------------
# Taxonomy retrieval behaviour
# --------------------------------------------------------------------------------


def test_repeated_signals_actually_outweigh_incidental_mentions():
    """Repetition in the query must change ranking, or the weighting is a no-op.

    `build_query` repeats breadcrumbs and any declared category to weight them. If
    retrieval de-duplicated query tokens, that repetition would cost nothing and buy
    nothing - a silent bug that leaves the intent in the code and the effect absent.
    """
    weighted = taxonomy.retrieve(taxonomy.build_query(breadcrumbs=["Store", "Bicycles"]), 40)
    diluted = taxonomy.retrieve("bicycles " + ("notebook paint kettle " * 12), 40)

    bike_rank = next(
        (i for i, c in enumerate(weighted) if "Bicycle" in c), len(weighted)
    )
    diluted_rank = next((i for i, c in enumerate(diluted) if "Bicycle" in c), len(diluted))
    assert bike_rank < diluted_rank


def test_retrieval_spans_unrelated_categories():
    """Retrieval must work across the taxonomy, not just the branches we sampled.

    All five supplied pages fall into apparel, home, or hardware. These do not.
    """
    expectations = [
        ("fresh whole milk dairy", "Food, Beverages & Tobacco"),
        ("acoustic guitar strings", "Arts & Entertainment"),
        ("dog collar leash", "Animals & Pet Supplies"),
        ("printer toner cartridge", "Electronics"),
    ]
    for query, expected_root in expectations:
        results = taxonomy.retrieve(query, 30)
        assert results, f"no candidates for {query!r}"
        assert any(
            entry.startswith(expected_root) for entry in results
        ), f"{query!r} retrieved nothing under {expected_root!r}: {results[:3]}"


def test_every_retrieved_candidate_is_verbatim():
    """Whatever the query, the shortlist must contain only real taxonomy entries.

    The model is instructed to copy verbatim from this list, so a malformed entry here
    becomes an invalid category downstream.
    """
    from models import VALID_CATEGORIES

    for query in ("garden hose", "infant car seat", "welding helmet", "sheet music"):
        for entry in taxonomy.retrieve(query, 25):
            assert entry in VALID_CATEGORIES


# --------------------------------------------------------------------------------
# Structural defences must not depend on page size or layout
# --------------------------------------------------------------------------------


def test_cross_sell_defence_on_a_differently_shaped_page():
    """An image linking to another product must be excluded regardless of layout.

    A single-item cross-sell is not a link farm by any density threshold, so the
    defence cannot rest on density alone. The rule that generalises is simpler: an
    image wrapped in a link to a different page depicts a different product.
    """
    html = """
    <html><head><link rel="canonical" href="https://shop.test/p/kettle"></head><body>
      <div><img src="https://cdn.test/i/kettle-main.jpg"></div>
      <aside>
        <a href="/p/toaster"><img src="https://cdn.test/i/toaster.jpg"></a>
      </aside>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/i/kettle-main.jpg" in urls
    assert all("toaster" not in url for url in urls)


def test_gallery_images_linking_to_the_same_page_are_kept():
    """A gallery image wrapped in a same-page zoom link must survive.

    The counterpart to the test above: excluding every linked image would gut ordinary
    galleries, which commonly wrap each thumbnail in a zoom anchor.
    """
    html = """
    <html><head><link rel="canonical" href="https://shop.test/p/kettle"></head><body>
      <a href="/p/kettle"><img src="https://cdn.test/i/kettle-1.jpg"></a>
      <a href="#zoom"><img src="https://cdn.test/i/kettle-2.jpg"></a>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert len(urls) == 2, urls


def test_deeply_nested_state_is_still_reachable():
    """Product records nested deep in a state tree must still be discovered.

    Redux-style stores nest heavily. A depth cap set too shallow would silently return
    nothing on exactly the pages that need the fallback most.
    """
    payload = {"name": "Ceramic Mug", "sku": "MUG-1", "price": 16.0}
    for key in reversed(["a", "b", "c", "d", "e", "f", "g"]):
        payload = {key: payload}

    import json

    html = f"<script>window.__S__ = {json.dumps(payload)};</script>"
    assert any("MUG-1" in str(s) for s in extract.run(html).json_subtrees)


def test_empty_and_malformed_pages_do_not_raise():
    """Degenerate inputs must return empty evidence, never an exception.

    At scale a fraction of fetched pages are error pages, redirects, or truncated
    HTML. Those must not be able to kill a batch.
    """
    for html in ("", "<html>", "not html at all", "<html><body></body></html>"):
        evidence = extract.run(html)
        assert isinstance(evidence.images, list)
        assert isinstance(evidence.text_blocks, list)
