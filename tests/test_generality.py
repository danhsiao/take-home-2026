"""Falsification tests: PDP shapes deliberately unlike the supplied pages.
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


def test_option_tables_survive_the_noise_filter():
    """A configurator modelled as questions and answers is product data, not Q&A.

    The noise vocabulary that shrinks oversized state blobs matches on key names, and
    "question" means customer Q&A on one page and the colour picker on the next.
    Dropping marketing bulk is recoverable; dropping the option table is not, because
    on a page with no JSON-LD the variant matrix exists nowhere else.
    """
    import json

    state = {
        "name": "Trail Backpack",
        "sku": "BP-1",
        "price": 89.0,
        "questions": [
            {
                "title": "Capacity",
                "answers": [
                    {"title": "30L", "skus": ["BP-30-RED", "BP-30-BLK"]},
                    {"title": "45L", "skus": ["BP-45-RED"]},
                ],
            }
        ],
        # Genuine Q&A bulk under the same vocabulary must still be dropped.
        "questionFeed": [{"body": "x" * 4000}],
    }
    subtrees = extract.run(f"<script>window.__S__ = {json.dumps(state)};</script>").json_subtrees
    blob = json.dumps(subtrees)
    assert "BP-30-RED" in blob, "option table was discarded as noise"
    assert "xxxx" not in blob, "review bulk should still be dropped"


def test_option_table_is_joined_into_real_combinations():
    """Groups joined by identifier give the true matrix, including its gaps.

    The combinations a page publishes are rarely the full cross-product. Joining on
    the identifier reproduces exactly the ones that exist: 30L comes in red and black,
    45L only in red, and no join ever invents 45L/black.
    """
    import json

    state = {
        "name": "Trail Backpack",
        "sku": "BP-1",
        "price": 89.0,
        "options": [
            {
                "title": "Capacity",
                "values": [
                    {"title": "30L", "skus": ["BP-30-RED", "BP-30-BLK"]},
                    {"title": "45L", "skus": ["BP-45-RED"]},
                ],
            },
            {
                "title": "Colour",
                "values": [
                    {"title": "Red", "skus": ["BP-30-RED", "BP-45-RED"]},
                    {"title": "Black", "skus": ["BP-30-BLK"]},
                ],
            },
        ],
    }
    blobs = extract._find_json_blobs(f"<script>window.__S__ = {json.dumps(state)};</script>")
    variants = extract.harvest_variant_graph(blobs)

    combinations = {(v.options.get("Capacity"), v.options.get("Colour")) for v in variants}
    assert combinations == {("30L", "Red"), ("30L", "Black"), ("45L", "Red")}
    assert all(v.sku for v in variants), "each joined combination is a real identifier"


def test_bare_itemprop_price_is_read_without_an_itemscope():
    """A buy box annotated with `itemprop` but no `itemscope` still states its price.

    Read strictly, an `itemprop` outside an item means nothing. Read practically, it is
    one of the commonest ways a price appears in machine-readable form, and a reader
    that insists on the letter of the spec throws the price away and then has nothing
    to fall back on but prose full of other products' prices.
    """
    html = """
    <html><head><link rel="canonical" href="https://shop.test/p/kettle"></head><body>
      <h1>Enamel Kettle</h1>
      <div class="buy-box">
        <span hidden itemprop="priceCurrency">EUR</span>
        <span itemprop="price">€42.00</span>
      </div>
    </body></html>
    """
    import pipeline

    facts = pipeline.derive_facts(extract.run(html))
    assert facts.price is not None, "a bare itemprop price must still be read"
    assert facts.price.price == pytest.approx(42.00)
    assert facts.price.currency == "EUR"


def test_overlay_links_mark_a_tile_as_another_product():
    """A tile made clickable by an overlay anchor is still another product's tile.

    Laying a transparent link over a card is a standard way to avoid nesting
    interactive content, and it puts the anchor *beside* the image rather than around
    it. A defence that only looks upwards from the image sees nothing at all.
    """
    html = """
    <html><head><link rel="canonical" href="https://shop.test/p/kettle">
    <meta property="og:image" content="https://cdn.test/shop/img/kettle.jpg">
    <meta property="og:title" content="Enamel Kettle"></head><body>
      <h1>Enamel Kettle</h1>
      <div><img src="https://cdn.test/shop/img/kettle.jpg"></div>
      <section>
        <div class="tile">
          <div><img src="https://cdn.test/shop/img/toaster.jpg"></div>
          <a class="overlay" href="/p/toaster">Toaster</a>
        </div>
      </section>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/kettle.jpg" in urls
    assert all("toaster" not in url for url in urls)


def test_help_links_beside_a_gallery_are_not_tiles():
    """Returns and delivery links sit beside every buy box and mean nothing.

    The counterpart to the test above, and the reason tiles are recognised by links
    into this page's own URL namespace rather than by outbound links in general: one
    real page puts `/returns` next to its gallery, and treating that as evidence of a
    tile discards the entire gallery.
    """
    html = """
    <html><head><link rel="canonical" href="https://shop.test/product/25289/kettle">
    <meta property="og:image" content="https://cdn.test/shop/img/kettle-1.jpg">
    <meta property="og:title" content="Enamel Kettle"></head><body>
      <h1>Enamel Kettle</h1>
      <section>
        <div><img src="https://cdn.test/shop/img/kettle-1.jpg"></div>
        <div><img src="https://cdn.test/shop/img/kettle-2.jpg"></div>
        <a href="/returns">Returns</a><a href="/measure-for-delivery">Delivery</a>
      </section>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/kettle-2.jpg" in urls, urls


def test_one_asset_served_under_two_routes_is_one_image():
    """A CDN that exposes the same file by slug and by id has not got two photographs.

    Content-addressed filenames make this decidable: the id and digest identify the
    bytes, whatever route leads to them. Without it the sharing URL and the gallery
    URL look like different images filed in different stores - which drops the gallery
    on any site that separates the two.
    """
    token = "55c9d30f-b3bc-45af-a2df-f78ab70d0df6.fbceb89fe250d24e4079259d68079645"
    html = f"""
    <html><head><link rel="canonical" href="https://shop.test/ip/kettle/123">
    <meta property="og:image" content="https://cdn.test/seo/Enamel-Kettle-1-5L_{token}.jpeg">
    <meta property="og:title" content="Enamel Kettle"></head><body>
      <h1>Enamel Kettle</h1>
      <div><img src="https://cdn.test/asr/{token}.jpeg"></div>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert len(urls) == 1, urls


def test_variant_graph_resolves_media_price_and_stock_references():
    """A SKU row pointing at media, price and stock tables must be followed.

    This is the relationship the whole feature rests on: without it a shopper picks a
    colour and sees the same photograph, because nothing ever connected the colour to
    its own imagery. The reference styles here - a list of media ids, a price id, and a
    nested stock record - are the common ones, and none of the key names is special.
    """
    import json

    state = {
        "name": "Enamel Kettle",
        "sku": "K-1",
        "options": [
            {
                "title": "Finish",
                "values": [
                    {"title": "Cream", "skus": ["K-CRM"]},
                    {"title": "Slate", "skus": ["K-SLT"]},
                ],
            },
            {
                "title": "Capacity",
                "values": [{"title": "1.5L", "skus": ["K-CRM", "K-SLT"]}],
            },
        ],
        "skus": [
            {
                "id": "K-CRM",
                "media": ["m-cream", "m-detail"],
                "price": "p-full",
                "availability": {"status": "In Stock"},
            },
            {
                "id": "K-SLT",
                "media": ["m-slate"],
                "price": "p-sale",
                "availability": {"status": "Out of Stock"},
            },
        ],
        "media": [
            {"id": "m-cream", "src": "https://cdn.test/shop/img/kettle-cream.jpg"},
            {"id": "m-slate", "src": "https://cdn.test/shop/img/kettle-slate.jpg"},
            {"id": "m-detail", "src": "https://cdn.test/shop/img/kettle-detail.jpg"},
        ],
        "prices": [{"id": "p-full", "amount": 59.0}, {"id": "p-sale", "amount": 44.0}],
    }
    blobs = extract._find_json_blobs(f"<script>window.__S__ = {json.dumps(state)};</script>")
    by_sku = {row.sku: row for row in extract.harvest_variant_graph(blobs)}

    cream, slate = by_sku["K-CRM"], by_sku["K-SLT"]
    assert cream.image_urls == [
        "https://cdn.test/shop/img/kettle-cream.jpg",
        "https://cdn.test/shop/img/kettle-detail.jpg",
    ]
    assert slate.image_urls == ["https://cdn.test/shop/img/kettle-slate.jpg"]
    assert (cream.amount, slate.amount) == (59.0, 44.0)
    assert (cream.available, slate.available) == (True, False)


def test_variant_graph_leaves_undeclared_stock_unknown():
    """A page that does not declare stock must yield None, never False.

    Reporting "unavailable" for a configuration the page simply never described is an
    invention, and one that would hide real products from a shopper.
    """
    import json

    state = {
        "name": "Enamel Kettle",
        "sku": "K-1",
        "options": [
            {"title": "Finish", "values": [{"title": "Cream", "skus": ["K-CRM"]}]},
            {"title": "Capacity", "values": [{"title": "1.5L", "skus": ["K-CRM"]}]},
        ],
        "skus": [{"id": "K-CRM", "price": "p-full"}],
        "prices": [{"id": "p-full", "amount": 59.0}],
    }
    blobs = extract._find_json_blobs(f"<script>window.__S__ = {json.dumps(state)};</script>")
    assert extract.harvest_variant_graph(blobs)[0].available is None


def test_option_table_needs_two_dimensions_to_join():
    """One group joins against nothing, so it must not be read as a matrix.

    Requiring an intersection is what stops an unrelated list of identifiers - a
    filter, a sort order, a size chart - from being presented as variants.
    """
    import json

    state = {
        "name": "Espresso Beans",
        "sku": "EB-1",
        "price": 14.0,
        "options": [{"title": "Grind", "values": [{"title": "Fine", "skus": ["EB-F"]}]}],
    }
    blobs = extract._find_json_blobs(f"<script>window.__S__ = {json.dumps(state)};</script>")
    assert extract.harvest_variant_graph(blobs) == []


def test_empty_and_malformed_pages_do_not_raise():
    """Degenerate inputs must return empty evidence, never an exception.

    At scale a fraction of fetched pages are error pages, redirects, or truncated
    HTML. Those must not be able to kill a batch.
    """
    for html in ("", "<html>", "not html at all", "<html><body></body></html>"):
        evidence = extract.run(html)
        assert isinstance(evidence.images, list)
        assert isinstance(evidence.text_blocks, list)


# --------------------------------------------------------------------------------
# Identity: which entity is this document about?
#
# The rules under test here are entailments of published specifications - RFC 6596
# defines the canonical link as the preferred URI for *this* resource, schema.org
# defines a differing SKU as a different product - so none of them may depend on a
# retailer, a CDN, or a category.
# --------------------------------------------------------------------------------


def test_embedded_state_describing_another_route_is_distrusted():
    """A payload that says it belongs to a different page must not supply evidence.

    Single-page-app frameworks serialise the state of the page that was *server*
    rendered, and client-side navigation never refreshes it. Save the page afterwards
    and the document is a product while its embedded state is a category listing full
    of entirely real, entirely different products. Nothing about those records' shape
    marks them as foreign - they have names, prices and photography - so shape-based
    extraction mines them happily.

    The same check covers redirects, interstitials and bot-walls.
    """
    import json

    stale = {
        "page": "/browse/[...slug]",
        "searchResult": {
            "items": [
                {
                    "name": "Chain Lubricant 100ml",
                    "canonicalUrl": "/p/chain-lubricant/88881111",
                    "images": ["https://cdn.bike.test/media/chain-lube.jpg"],
                },
                {
                    "name": "Track Pump",
                    "canonicalUrl": "/p/track-pump/88882222",
                    "images": ["https://cdn.bike.test/media/track-pump.jpg"],
                },
            ]
        },
    }
    html = f"""
    <html><head>
      <link rel="canonical" href="https://bike.test/p/road-helmet/99990000">
      <meta property="og:image" content="https://cdn.bike.test/media/helmet-hero.jpg">
      <title>Aero Road Helmet</title>
    </head><body>
      <h1>Aero Road Helmet</h1>
      <script type="application/json">{json.dumps(stale)}</script>
    </body></html>
    """
    evidence = extract.run(html)

    assert evidence.stale_blobs == 1
    urls = " ".join(asset.url for asset in evidence.images)
    assert "chain-lube" not in urls
    assert "track-pump" not in urls


def test_a_payload_that_declares_no_route_is_still_trusted():
    """Silence is not disagreement.

    Most embedded state says nothing about which page it belongs to. Treating that
    silence as a mismatch would discard the evidence on the majority of pages, so the
    check must fire only on an actual contradiction.
    """
    import json

    state = {
        "name": "Hex Key Set",
        "sku": "HK-9",
        "price": 18.0,
        "images": ["https://cdn.supply.test/assets/hex-keys.jpg"],
    }
    html = f"""
    <html><head><link rel="canonical" href="https://supply.test/p/hex-key-set/4455"></head>
    <body><h1>Hex Key Set</h1>
      <script type="application/json">{json.dumps(state)}</script>
    </body></html>
    """
    evidence = extract.run(html)
    assert evidence.stale_blobs == 0
    assert any("hex-keys" in asset.url for asset in evidence.images)


def test_gallery_on_a_different_route_from_the_hero_survives():
    """A merchant may share one route and serve the gallery from another.

    Comparing an image's asset root against the OpenGraph anchor alone discards the
    entire real gallery whenever those differ - while admitting any cross-sell
    thumbnail that happens to share the sharing route. That filter does not weaken, it
    inverts, which is worse than having no filter at all.

    The product's own rendered region is the second witness that prevents it.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://books.test/p/atlas-of-moss/71625">
      <meta property="og:title" content="Atlas of Moss">
      <meta property="og:image" content="https://img.books.test/share/atlas-of-moss-cover.jpg">
    </head><body>
      <main><h1>Atlas of Moss</h1>
        <img src="https://img.books.test/deliver/6f2a9c41-plate-one.jpg">
        <img src="https://img.books.test/deliver/6f2a9c41-plate-two.jpg">
      </main>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any("plate-one" in url for url in urls)
    assert any("plate-two" in url for url in urls)


def test_cross_sell_sharing_the_product_cdn_is_still_dropped():
    """Ownership is a property of the entity graph, not of a URL's shape.

    A recommendation tile is served from the same host and the same asset root as the
    product's own photography, at the same resolution. No URL-shape rule can separate
    them. The tile declares which product it advertises, and that identifier is not
    one this page claims.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://cosmetic.test/p/rose-serum/30301111">
      <meta property="og:title" content="Rose Serum">
      <meta property="og:image" content="https://cdn.cosmetic.test/media/rose-serum-hero.jpg">
    </head><body>
      <main><h1>Rose Serum</h1>
        <img src="https://cdn.cosmetic.test/media/rose-serum-detail.jpg">
      </main>
      <section>
        <div data-item-id="30302222">
          <img src="https://cdn.cosmetic.test/media/clay-mask.jpg">
        </div>
      </section>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any("rose-serum-detail" in url for url in urls)
    assert all("clay-mask" not in url for url in urls)


def test_an_analytics_attribute_is_not_a_product_identifier():
    """Instrumentation ids must not be mistaken for merchandise ids.

    Gallery controls routinely carry beacon and experiment identifiers. Reading any
    attribute whose name ends in "id" as a product id lets one tracking attribute on
    the gallery's own button condemn the entire real gallery as somebody else's.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://supply.test/p/torque-wrench/50607080">
      <meta property="og:title" content="Torque Wrench">
      <meta property="og:image" content="https://cdn.supply.test/media/wrench-hero.jpg">
    </head><body>
      <main><h1>Torque Wrench</h1>
        <button data-dca-aid="B:8817FE20AC" data-testid="hero-button">
          <img src="https://cdn.supply.test/media/wrench-detail.jpg">
        </button>
      </main>
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any("wrench-detail" in url for url in urls)


# --------------------------------------------------------------------------------
# Rendered controls
# --------------------------------------------------------------------------------


def test_aria_options_are_read_when_no_product_json_exists():
    """A client-rendered PDP keeps its options only in the accessibility tree.

    WAI-ARIA defines `radiogroup`/`radio` and the `aria-checked` state, so a merchant
    using them is making a machine-readable statement about a set of mutually
    exclusive choices. On a page whose options were fetched after render, that
    statement is the only surviving record that they exist.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://grocer.test/p/loose-leaf-tea/60123">
      <meta property="og:title" content="Loose Leaf Tea">
    </head><body>
      <main><h1>Loose Leaf Tea</h1>
        <div role="radiogroup" aria-label="Roast">
          <div role="radio" aria-checked="true">
            <span style="background-image: url('https://cdn.grocer.test/s/light.jpg')"></span>
            Light
          </div>
          <div role="radio" aria-checked="false">
            <span style="background-image: url('https://cdn.grocer.test/s/dark.jpg')"></span>
            Dark
          </div>
        </div>
      </main>
    </body></html>
    """
    groups = extract.run(html).option_groups
    roast = next(group for group in groups if group.name == "Roast")
    assert [value.label for value in roast.values] == ["Light", "Dark"]
    assert roast.values[0].selected is True
    assert roast.values[1].image_url == "https://cdn.grocer.test/s/dark.jpg"


def test_a_composed_accessible_name_keeps_only_its_leading_field():
    """A swatch that announces its price must not carry the price into the label.

    Packing label and annotations into one comma-delimited announcement is correct
    accessibility practice - a comma is how assistive technology is cued to pause
    between fields - so the first field is the element's own label. Reading the whole
    string would put a currency amount inside a dimension value.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://grocer.test/p/olive-oil-tin/70123">
      <meta property="og:title" content="Olive Oil Tin">
    </head><body>
      <main><h1>Olive Oil Tin</h1>
        <div role="radiogroup" aria-label="Size">
          <div role="radio" aria-checked="true">
            <span>500ml, $12.00, was $15.00</span>
            <img src="https://cdn.grocer.test/s/500ml.jpg">
          </div>
          <div role="radio"><span>1L, $20.00, Out of stock</span>
            <img src="https://cdn.grocer.test/s/1l.jpg"></div>
        </div>
      </main>
    </body></html>
    """
    groups = extract.run(html).option_groups
    size = next(group for group in groups if group.name == "Size")
    assert [value.label for value in size.values] == ["500ml", "1L"]


def test_a_variant_swatch_strip_is_not_treated_as_a_link_farm():
    """Where each variant is its own page, the swatch strip *is* a rail of links.

    Density-based cross-sell suppression condemns exactly that shape - one link per
    swatch, almost no prose. An explicit ARIA declaration has to outrank an inferred
    heuristic, or the control we most need is the one we always discard.
    """
    swatches = "".join(
        f'<div role="radio"><a href="/p/paint/900{n}">'
        f'<img src="https://cdn.paint.test/s/{n}.jpg" alt="Shade {n}"></a></div>'
        for n in range(8)
    )
    html = f"""
    <html><head>
      <link rel="canonical" href="https://paint.test/p/paint/9000">
      <meta property="og:title" content="Wall Paint">
    </head><body>
      <main><h1>Wall Paint</h1>
        <div role="radiogroup" aria-label="Shade">{swatches}</div>
      </main>
    </body></html>
    """
    groups = extract.run(html).option_groups
    shade = next(group for group in groups if group.name == "Shade")
    assert len(shade.values) == 8


def test_a_disabled_placeholder_option_is_not_a_choice():
    """HTML's way of rendering an unsubmittable prompt is a disabled leading option.

    Keeping it would also lend the group a false availability signal, since a disabled
    option is indistinguishable from an out-of-stock configuration.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://supply.test/p/drill-bit-set/12321">
      <meta property="og:title" content="Drill Bit Set">
    </head><body>
      <main><h1>Drill Bit Set</h1>
        <select aria-label="Shank">
          <option disabled selected>Choose a shank</option>
          <option>Hex</option>
          <option>Round</option>
        </select>
      </main>
    </body></html>
    """
    groups = extract.run(html).option_groups
    shank = next(group for group in groups if group.name == "Shank")
    assert [value.label for value in shank.values] == ["Hex", "Round"]
    assert all(value.available is None for value in shank.values)


def test_form_fields_do_not_become_product_variants():
    """A lead-capture form is a choice set too, and must not be read as merchandise.

    No structural test tells "Colour" from "Where will the work be done?" by name
    without encoding a vocabulary that breaks in the next language. What separates
    them is payload: a product option carries something specific to that configuration
    - its own photograph, its own page, or a statement that it is unavailable. A list
    of place names carries none of those.
    """
    import pipeline

    states = "".join(f"<option>Region {n}</option>" for n in range(30))
    html = f"""
    <html><head>
      <link rel="canonical" href="https://supply.test/p/ladder/33445">
      <meta property="og:title" content="Step Ladder">
    </head><body>
      <main><h1>Step Ladder</h1>
        <select aria-label="Where will the work be done?">{states}</select>
      </main>
    </body></html>
    """
    evidence = extract.run(html)
    assert evidence.option_groups  # harvested as evidence
    assert pipeline._variants_from_controls(evidence) == []  # but never merchandise


def test_control_variants_never_form_a_cartesian_product():
    """Rendered controls state dimensions and values, never which combinations exist.

    Two groups of six could be six configurations or thirty-six, and the page does not
    say. Enumerating one group while pinning the others to their displayed selection
    keeps every emitted variant one the page actually shows.
    """
    import pipeline

    shades = "".join(
        f'<div role="radio"><img src="https://cdn.paint.test/s/{n}.jpg" alt="Shade {n}"></div>'
        for n in range(6)
    )
    finishes = "".join(
        f'<div role="radio" {"aria-checked=\"true\"" if n == 0 else ""}>'
        f'<img src="https://cdn.paint.test/f/{n}.jpg" alt="Finish {n}"></div>'
        for n in range(6)
    )
    html = f"""
    <html><head>
      <link rel="canonical" href="https://paint.test/p/emulsion/7788">
      <meta property="og:title" content="Emulsion">
    </head><body>
      <main><h1>Emulsion</h1>
        <div role="radiogroup" aria-label="Shade">{shades}</div>
        <div role="radiogroup" aria-label="Finish">{finishes}</div>
      </main>
    </body></html>
    """
    variants = pipeline._variants_from_controls(extract.run(html))
    assert 0 < len(variants) <= 6  # never 36


# --------------------------------------------------------------------------------
# Renditions: emitting the size the merchant's own site shows
# --------------------------------------------------------------------------------


def test_a_bare_numbered_rendition_picks_the_largest():
    """Some CDNs encode the size as a plain trailing number and nothing else.

    A bare number is not treated as a size when normalising a URL, because in
    isolation it is more often an asset id. The ambiguity is settled by the page: these
    URLs are already known to be one asset, so a number that varies while the identity
    stays fixed is a rendition. Without this the group reports an unknown width and
    selection falls through to the arbitrary tiebreak - the shortest URL, which is
    always the smallest rendition.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://tools.test/p/torque-wrench/50607080">
      <meta property="og:title" content="Torque Wrench">
    </head><body><main><h1>Torque Wrench</h1>
      <img src="https://cdn.tools.test/i/6f2a9c41-0d3b-4d02-9a1e-11b2c3d4e5f6/wrench_100.jpg">
      <img src="https://cdn.tools.test/i/6f2a9c41-0d3b-4d02-9a1e-11b2c3d4e5f6/wrench_1000.jpg">
    </main></body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any(url.endswith("wrench_1000.jpg") for url in urls)
    assert not any(url.endswith("wrench_100.jpg") for url in urls)


def test_consecutive_numbers_are_not_treated_as_renditions():
    """`_10`, `_11`, `_12` enumerate something; they do not scale it.

    Renditions of one photograph are produced at meaningfully different scales, so the
    largest is at least twice the smallest. Numbers sitting next to each other are a
    sequence, and collapsing them would silently merge distinct photographs.
    """
    import images

    best = {"k": ((0, 0, 0), "https://cdn.test/a_10.jpg", None, 0)}
    siblings = {"k": ["https://cdn.test/a_10.jpg", "https://cdn.test/a_11.jpg"]}
    resolved = images._prefer_larger_numbered_rendition(best, siblings)
    assert resolved["k"][1] == "https://cdn.test/a_10.jpg"


def test_a_published_size_template_upgrades_a_thumbnail():
    """A page may render thumbnails and publish the full-size pattern separately.

    When JavaScript builds the real gallery URL at runtime, the saved HTML contains
    only thumbnails - but the merchant often also publishes the pattern with the size
    slot left as an explicit placeholder. Both halves then come from the page: the
    pattern it declares, and a size it is seen to use for the same product.
    """
    import json

    state = {
        "media": {
            "images": [
                {"url": "https://cdn.books.test/i/0a1b2c3d4e5f60718293a4b5c6d7e8f9/plate_<SIZE>.jpg"}
            ]
        }
    }
    html = f"""
    <html><head>
      <link rel="canonical" href="https://books.test/p/atlas-of-moss/71625">
      <meta property="og:title" content="Atlas of Moss">
    </head><body><main><h1>Atlas of Moss</h1>
      <img src="https://cdn.books.test/i/0a1b2c3d4e5f60718293a4b5c6d7e8f9/plate_100.jpg">
      <img src="https://cdn.books.test/i/ffeeddccbbaa99887766554433221100/cover_1000.jpg">
      <script type="application/json">{json.dumps(state)}</script>
    </main></body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any(url.endswith("plate_1000.jpg") for url in urls), urls
    assert not any(url.endswith("plate_100.jpg") for url in urls)


def test_a_template_never_introduces_a_new_photograph():
    """Templates may restate an accepted image; they may not add one.

    A resolved template is admitted only when it is a rendition of an image the
    harvesting rules already kept. That keeps this change strictly about *which
    rendition* is emitted - it cannot reintroduce a cross-sell image that ownership
    filtering rejected, on this page or any other.
    """
    import json

    state = {
        "recommendations": [
            {"url": "https://cdn.books.test/i/aaaabbbbccccddddeeeeffff00001111/other_<SIZE>.jpg"}
        ]
    }
    html = f"""
    <html><head>
      <link rel="canonical" href="https://books.test/p/atlas-of-moss/71625">
      <meta property="og:title" content="Atlas of Moss">
    </head><body><main><h1>Atlas of Moss</h1>
      <img src="https://cdn.books.test/i/ffeeddccbbaa99887766554433221100/cover_1000.jpg">
      <script type="application/json">{json.dumps(state)}</script>
    </main></body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert all("other_" not in url for url in urls), urls


def test_preloaded_renditions_are_harvested_without_an_as_attribute():
    """`as="image"` is a fetch-priority hint, not what makes an href an image.

    A page that preloads its full-size gallery while rendering thumbnails in markup
    would otherwise be read as having only thumbnails.
    """
    html = """
    <html><head>
      <link rel="canonical" href="https://grocer.test/p/olive-oil/60123">
      <meta property="og:title" content="Olive Oil">
      <link rel="preload" href="https://cdn.grocer.test/i/7c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f/oil_1200.jpg">
    </head><body><main><h1>Olive Oil</h1>
      <img src="https://cdn.grocer.test/i/7c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f/oil_100.jpg">
    </main></body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert any(url.endswith("oil_1200.jpg") for url in urls), urls
