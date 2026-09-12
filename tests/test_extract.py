"""Unit tests for deterministic extraction.
"""

import extract
import images
import taxonomy


# --------------------------------------------------------------------------------
# JSON-LD shapes
# --------------------------------------------------------------------------------


def test_parses_jsonld_graph_container():
    """`@graph` is a standard container and must be flattened into typed items."""
    html = """
    <script type="application/ld+json">
    {"@context":"https://schema.org","@graph":[
      {"@type":"Product","name":"Widget"},
      {"@type":"BreadcrumbList","itemListElement":[{"name":"Home"},{"name":"Widgets"}]}
    ]}
    </script>
    """
    evidence = extract.run(html)
    assert len(extract.find_typed(evidence.structured, "Product")) == 1
    assert evidence.breadcrumbs == ["Home", "Widgets"]


def test_parses_jsonld_array_at_top_level():
    """A bare array of items is equally valid JSON-LD."""
    html = """
    <script type="application/ld+json">
    [{"@type":"Product","name":"A"},{"@type":"Product","name":"B"}]
    </script>
    """
    assert len(extract.find_typed(extract.run(html).structured, "Product")) == 2


def test_flattens_product_group_variants():
    """A ProductGroup's `hasVariant` Products must surface as items in their own right."""
    html = """
    <script type="application/ld+json">
    {"@type":"ProductGroup","name":"Shirt","variesBy":["https://schema.org/size"],
     "hasVariant":[{"@type":"Product","name":"Shirt S","size":"S"},
                   {"@type":"Product","name":"Shirt M","size":"M"}]}
    </script>
    """
    products = extract.find_typed(extract.run(html).structured, "Product")
    assert {p["size"] for p in products} == {"S", "M"}


def test_accepts_schema_org_url_types():
    """`@type` may be a full URL rather than a bare name."""
    html = '<script type="application/ld+json">{"@type":"https://schema.org/Product","name":"X"}</script>'
    assert len(extract.find_typed(extract.run(html).structured, "Product")) == 1


def test_malformed_jsonld_does_not_break_extraction():
    """A broken block elsewhere on the page must not fail the whole page."""
    html = """
    <script type="application/ld+json">{ this is not json </script>
    <script type="application/ld+json">{"@type":"Product","name":"Survivor"}</script>
    """
    products = extract.find_typed(extract.run(html).structured, "Product")
    assert [p["name"] for p in products] == ["Survivor"]


# --------------------------------------------------------------------------------
# Embedded application state
# --------------------------------------------------------------------------------


def test_discovers_product_shaped_subtree_in_window_assignment():
    """Product records in framework state must be found by shape, not by key path."""
    html = """
    <script>window.__ANY_NAME__ = {"deeply":{"nested":{"thing":
      {"name":"Deep Widget","price":42.5,"sku":"DW-1"}}}};</script>
    """
    subtrees = extract.run(html).json_subtrees
    assert any(s.get("sku") == "DW-1" for s in subtrees), subtrees


def test_brace_matching_survives_braces_inside_strings():
    """Braces and escapes inside string literals must not confuse the depth count."""
    html = """
    <script>window.__S__ = {"label":"a } weird { string \\" with escapes",
      "name":"Tricky","price":9.99};</script>
    """
    subtrees = extract.run(html).json_subtrees
    assert any(s.get("name") == "Tricky" for s in subtrees)


def test_ignores_non_product_state():
    """State with no product shape must not be forwarded as evidence."""
    html = '<script>window.__CONFIG__ = {"locale":"en","debug":true,"retries":3};</script>'
    assert extract.run(html).json_subtrees == []


def test_denoises_bulky_non_product_keys():
    """Reviews and analytics must be stripped, while product structure survives."""
    html = """
    <script>window.__S__ = {"name":"Thing","sku":"T-1","price":10,
      "reviews":[{"text":"%s"}],"analyticsPayload":{"junk":"%s"}};</script>
    """ % ("x" * 3000, "y" * 3000)
    subtrees = extract.run(html).json_subtrees
    assert subtrees, "product subtree should survive denoising"
    blob = str(subtrees)
    assert "T-1" in blob
    assert "xxxx" not in blob and "yyyy" not in blob


# --------------------------------------------------------------------------------
# Cross-sell suppression
# --------------------------------------------------------------------------------


def test_link_dense_region_is_suppressed():
    """A recommendation strip is high link density and must not reach the evidence."""
    html = """
    <html><head><meta property="og:title" content="Real Product">
    <link rel="canonical" href="https://x.test/p/real"></head><body>
      <div><h1>Real Product</h1><p>Genuine description of the real product.</p></div>
      <ul>
        <li><a href="/p/a">Alpha Thing</a></li>
        <li><a href="/p/b">Beta Thing</a></li>
        <li><a href="/p/c">Gamma Thing</a></li>
      </ul>
    </body></html>
    """
    blob = "\n".join(extract.run(html).text_blocks)
    assert "Genuine description" in blob
    assert "Alpha Thing" not in blob


def test_prose_with_a_few_links_is_kept():
    """Low link density means content, even when links are present.

    The counterpart to the test above: suppressing on link *count* alone would remove
    ordinary product copy that happens to link to a size guide.
    """
    html = """
    <html><head><meta property="og:title" content="Real Product">
    <link rel="canonical" href="https://x.test/p/real"></head><body>
      <div><h1>Real Product</h1>
      <p>This is a long and genuine product description that goes on for a while
         about the materials and construction of the item, and happens to mention
         a <a href="/size-guide">size guide</a> and a <a href="/returns">returns
         policy</a> and our <a href="/shipping">shipping</a> terms in passing.</p>
      </div>
    </body></html>
    """
    blob = "\n".join(extract.run(html).text_blocks)
    assert "materials and construction" in blob


# --------------------------------------------------------------------------------
# Images: resolution grouping and selection
# --------------------------------------------------------------------------------


def test_srcset_with_relative_candidates():
    """Relative srcset candidates must parse - not every srcset holds absolute URLs."""
    parsed = images._parse_srcset("foo-320.jpg 320w, foo-640.jpg 640w")
    assert parsed == [("foo-320.jpg", 320), ("foo-640.jpg", 640)]


def test_srcset_with_commas_inside_urls():
    """CDN transform lists contain commas; splitting on every comma corrupts them."""
    value = (
        "https://cdn.test/a_1,b_2/img.jpg 800w, "
        "https://cdn.test/a_1,b_2/img.jpg 1600w"
    )
    parsed = images._parse_srcset(value)
    assert [w for _, w in parsed] == [800, 1600]
    assert all(url.startswith("https://cdn.test/a_1,b_2/") for url, _ in parsed)


def test_srcset_density_descriptors_carry_no_width():
    """`2x` is a pixel-density ratio, not a width, and must not be read as one."""
    parsed = images._parse_srcset("a.jpg 1x, b.jpg 2x")
    assert parsed == [("a.jpg", None), ("b.jpg", None)]


def test_groups_renditions_and_keeps_the_largest():
    """The same photo at several widths is one asset, emitted at its largest."""
    html = """
    <html><head><link rel="canonical" href="https://x.test/p"></head><body><div>
      <img srcset="https://cdn.test/i/photo.jpg?w=320 320w,
                   https://cdn.test/i/photo.jpg?w=2000 2000w">
    </div></body></html>
    """
    assets = extract.run(html).images
    assert len(assets) == 1
    assert assets[0].width == 2000
    # The emitted URL must be one the page really exposed, not a reconstruction.
    assert assets[0].url == "https://cdn.test/i/photo.jpg?w=2000"


def test_dimension_path_segments_group_together():
    """`/200x0/` and `/2000x1500/` of one image are renditions, not distinct assets."""
    a = images._asset_key("https://cdn.test/p/200x0/img.jpg")
    b = images._asset_key("https://cdn.test/p/2000x1500/img.jpg")
    assert a == b


def test_query_width_overrides_path_dimension():
    """A downscaling query parameter defeats the source dimensions in the path.

    `/2890x1500/photo.jpg?w=320` serves a 320px image. Reading the largest number in
    the URL would rank a thumbnail as the best available rendition.
    """
    assert images._effective_width("https://c.test/p/2890x1500/a.jpg?w=320") == 320


def test_relative_scale_transforms_are_not_widths():
    """`w_1.0` is a ratio in a transform chain, not a one-pixel width."""
    assert images._effective_width("https://c.test/t/c_scale,w_1.0/a.jpg") is None
    assert images._effective_width("https://c.test/t/w_1536/a.jpg") == 1536


def test_size_words_rank_when_no_width_is_declared():
    """With no pixel width anywhere, `-max` must beat `-thumb`."""
    assert images._size_word_rank("https://c.test/a-max.jpg") > images._size_word_rank(
        "https://c.test/a-thumb.jpg"
    )


def test_bare_numeric_suffix_is_not_treated_as_a_size():
    """`-1234.jpg` is far more often an asset id than a resolution.

    Stripping it would merge genuinely different photographs into one asset and
    silently lose images.
    """
    a = images._asset_key("https://cdn.test/i/photo-154290.jpg")
    b = images._asset_key("https://cdn.test/i/photo-154291.jpg")
    assert a != b


def test_extensionless_image_urls_are_found_in_state():
    """Image servers often serve from extensionless paths under an image-ish key."""
    html = """
    <script>window.__S__ = {"name":"Thing","sku":"T1","price":5,
      "media":[{"src":"https://img.test/is/image/wim/12345_0_44"}]};</script>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://img.test/is/image/wim/12345_0_44" in urls


def test_protocol_relative_urls_are_absolutised():
    """`//cdn/...` must become absolute, since validation compares exact strings."""
    html = """
    <html><head><link rel="canonical" href="https://x.test/p">
    <script type="application/ld+json">
    {"@type":"Product","name":"P","image":"//cdn.test/a.jpg"}</script>
    </head><body></body></html>
    """
    assert extract.run(html).images[0].url == "https://cdn.test/a.jpg"


def test_chrome_images_are_filtered_from_untrusted_sources():
    """Logos and icons in markup are not product photography."""
    html = """
    <html><head><link rel="canonical" href="https://x.test/p"></head><body><div>
      <img src="https://cdn.test/assets/logo.png">
      <img src="https://cdn.test/i/product.jpg">
    </div></body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/i/product.jpg" in urls
    assert all("logo" not in url for url in urls)


# --------------------------------------------------------------------------------
# Images: relevance filtering
# --------------------------------------------------------------------------------


def _page(body: str, anchor: str = "https://cdn.test/shop/img/9000_0_1.jpg") -> str:
    """A PDP whose merchant-nominated product image is `anchor`."""
    return f"""
    <html><head><link rel="canonical" href="https://x.test/p">
    <meta property="og:image" content="{anchor}">
    </head><body>{body}</body></html>
    """


def test_svg_assets_are_not_product_photography():
    """Interface chrome is shipped as SVG; product photography never is."""
    html = _page("""
      <img src="https://cdn.test/shop/img/heart-inactive.svg">
      <img src="https://cdn.test/shop/img/9000_0_2.jpg">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/9000_0_2.jpg" in urls
    assert all(not url.endswith(".svg") for url in urls)


def test_tiny_declared_renditions_are_dropped():
    """A 65px rendition is a swatch, not a gallery image."""
    html = _page("""
      <img src="https://cdn.test/shop/img/9000_0_2.jpg?wid=65">
      <img src="https://cdn.test/shop/img/9000_0_3.jpg?wid=1200">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/9000_0_3.jpg?wid=1200" in urls
    assert all("wid=65" not in url for url in urls)


def test_an_asset_offering_both_sizes_keeps_the_large_one():
    """The size gate runs after resolution selection, never before it."""
    html = _page("""
      <img src="https://cdn.test/shop/img/9000_0_2.jpg?wid=65">
      <img src="https://cdn.test/shop/img/9000_0_2.jpg?wid=1200">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/9000_0_2.jpg?wid=1200" in urls


def test_bounding_parameters_are_read_only_for_filtering():
    """`?max=` identifies a thumbnail, but must not outrank an unbounded original.

    Reading it as a width would make the bounded rendition win selection, and the
    full-resolution URL - whose width the page never states - would be discarded.
    """
    html = _page("""
      <img src="https://cdn.test/shop/img/9000_0_2.jpg">
      <img src="https://cdn.test/shop/img/9000_0_2.jpg?max=100">
      <img src="https://cdn.test/shop/img/9000_0_9.jpg?max=100">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/9000_0_2.jpg" in urls
    assert all("9000_0_9" not in url for url in urls)


def test_images_from_another_asset_root_are_dropped():
    """Marketing banners and UI assets live outside the product image store."""
    html = _page("""
      <img src="https://cdn.test/shop/img/9000_0_2.jpg">
      <img src="https://cdn.test/marketing/banners/free-shipping-truck.png">
      <img src="https://reviews.other.test/Product/1/2/square.jpg">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert urls == [
        "https://cdn.test/shop/img/9000_0_1.jpg",
        "https://cdn.test/shop/img/9000_0_2.jpg",
    ]


def test_another_configurations_photography_is_dropped():
    """Same host, same folder, full resolution - but a different item number.

    In practice this is the same photography re-registered under a neighbouring
    configuration's id. The gallery presents one configuration; the variant rows keep
    the association for the others.
    """
    html = _page("""
      <img src="https://cdn.test/shop/img/9000_0_2.jpg">
      <img src="https://cdn.test/shop/img/9001_0_1.jpg">
      <img src="https://cdn.test/shop/img/9001_0_2.jpg">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/9000_0_2.jpg" in urls
    assert all("/9001_" not in url for url in urls)


def test_ids_of_a_different_shape_are_left_alone():
    """The id gate only compares like with like; it is not a numeric prefix ban."""
    html = _page("""
      <img src="https://cdn.test/shop/img/12_detail.jpg">
      <img src="https://cdn.test/shop/img/photo-900.jpg">
    """)
    urls = [asset.url for asset in extract.run(html).images]
    assert "https://cdn.test/shop/img/12_detail.jpg" in urls
    assert "https://cdn.test/shop/img/photo-900.jpg" in urls


def test_pages_without_a_nominated_image_are_not_filtered_by_family():
    """With no anchor there is no reference point, so recall must not be narrowed."""
    html = """
    <html><head><link rel="canonical" href="https://x.test/p"></head><body>
      <img src="https://cdn.test/shop/img/9000_0_2.jpg">
      <img src="https://other.test/whatever/photo.jpg">
    </body></html>
    """
    urls = [asset.url for asset in extract.run(html).images]
    assert len(urls) == 2


def test_filtering_never_empties_a_gallery():
    """If every gate rejects everything, the unfiltered harvest is returned.

    A page we cannot reason about should degrade to raw recall, not to no images.
    """
    anchor = "https://cdn.test/shop/img/9000_0_1.jpg?wid=65"
    html = _page(
        '<img src="https://elsewhere.test/x/y/z.jpg?wid=20">', anchor=anchor
    )
    assert extract.run(html).images


# --------------------------------------------------------------------------------
# Taxonomy retrieval
# --------------------------------------------------------------------------------


def test_retrieval_finds_expected_leaf():
    """Recall@K is directly measurable, with no model involved."""
    results = taxonomy.retrieve("cordless drill power tool", 30)
    assert "Hardware > Tools > Drills > Handheld Power Drills" in results


def test_retrieval_returns_only_verbatim_entries():
    """Every candidate offered to the model must be a real taxonomy entry."""
    for entry in taxonomy.retrieve("floor lamp lighting", 20):
        assert taxonomy.is_valid(entry)


def test_retrieval_handles_empty_query():
    """An unusable query returns nothing rather than raising."""
    assert taxonomy.retrieve("", 10) == []


def test_query_builder_drops_site_name_crumb():
    """The first breadcrumb is the site name and is noise for classification."""
    query = taxonomy.build_query(breadcrumbs=["SomeStore", "Lighting", "Lamps"])
    assert "SomeStore" not in query
    assert "Lamps" in query


def test_query_builder_caps_description_length():
    """A bulleted spec list must not drown out the product type.

    Sentence-splitting fails here because specification lists contain no sentence
    boundary, so the cap is by characters.
    """
    query = taxonomy.build_query(name="Lamp", description="x" * 5000)
    assert len(query) < 1000
