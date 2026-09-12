"""Adversarial tests for the runtime validator.
"""

import pytest

from models import ImageAsset, PageEvidence, Price, ProductCandidate, Variant
from validate import validate_candidate


def codes(issues) -> set[str]:
    """Collapse issues to their codes, which is what tests assert on."""
    return {issue.code for issue in issues}


@pytest.fixture
def evidence() -> PageEvidence:
    """A small, realistic page: one product, two real variants, two images.

    The two variants that genuinely exist are Black/Small and Blue/Large. Note that
    "Black", "Blue", "Small" and "Large" all appear *somewhere* in this evidence - so
    a validator that only checked "do these words exist on the page" would happily
    accept Black/Large, which the page never offers. That is the trap the grounding
    rule is built to catch.
    """
    return PageEvidence(
        url="https://example.test/p/1",
        structured=[
            {
                "@type": "Product",
                "name": "Test Chair",
                "brand": {"name": "Testco"},
                "offers": {"price": 249.0, "priceCurrency": "USD"},
            }
        ],
        json_subtrees=[
            {"name": "Test Chair", "sku": "CH-BLK-S", "color": "Black", "size": "Small"},
            {"name": "Test Chair", "sku": "CH-BLU-L", "color": "Blue", "size": "Large"},
        ],
        text_blocks=["Test Chair", "$249.00", "Available in Black and Blue"],
        images=[
            ImageAsset(url="https://cdn.test/img/a.jpg", asset_key="cdn.test/img/a.jpg"),
            ImageAsset(url="https://cdn.test/img/b.jpg", asset_key="cdn.test/img/b.jpg"),
        ],
        meta={"og:title": "Test Chair"},
    )


def _candidate(**overrides) -> ProductCandidate:
    """A candidate that passes validation, so each test can break exactly one thing."""
    defaults = dict(
        name="Test Chair",
        brand="Testco",
        description="A chair for testing.",
        price=Price(price=249.0, currency="USD"),
        category="Furniture > Chairs",
        variants=[
            Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-BLK-S"),
            Variant(options={"Color": "Blue", "Size": "Large"}, sku="CH-BLU-L"),
        ],
    )
    defaults.update(overrides)
    return ProductCandidate(**defaults)


def test_accepts_a_fully_grounded_candidate(evidence):
    """The baseline: a candidate backed by evidence raises no errors.

    Without this, every other test could pass for the wrong reason - a validator that
    rejected everything would satisfy all the negative cases below.
    """
    issues = validate_candidate(_candidate(), evidence)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == [], f"unexpected errors: {errors}"


def test_does_not_create_cartesian_variants(evidence):
    """Combinations the page never offers must be rejected as ungrounded.

    Black/Large and Blue/Small are the Cartesian filler between the two real variants.
    Every individual option value is present in the evidence; only the *pairings* are
    invented. A count-based check cannot see this, which is why grounding tests
    co-occurrence within a single evidence source instead.
    """
    invented = _candidate(
        variants=[
            Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-BLK-S"),
            Variant(options={"Color": "Blue", "Size": "Large"}, sku="CH-BLU-L"),
            Variant(options={"Color": "Black", "Size": "Large"}),  # never offered
            Variant(options={"Color": "Blue", "Size": "Small"}),  # never offered
        ]
    )
    issues = validate_candidate(invented, evidence)

    ungrounded = [i for i in issues if i.code == "variant.ungrounded"]
    assert len(ungrounded) == 2, f"expected the 2 invented pairings, got {ungrounded}"
    # And crucially, the two real variants must survive.
    assert "variants[0]" not in {i.field for i in ungrounded}
    assert "variants[1]" not in {i.field for i in ungrounded}


def test_accepts_full_matrix_when_every_combination_is_evidenced(evidence):
    """A genuine full matrix must NOT be flagged just because it is complete.

    This is the counterpart to the test above and the reason grounding replaced
    counting: a page really can offer every colour in every size. Here all four
    combinations appear as real records, so all four are legitimate.
    """
    evidence.json_subtrees = [
        {"name": "Test Chair", "sku": "CH-BLK-S", "color": "Black", "size": "Small"},
        {"name": "Test Chair", "sku": "CH-BLK-L", "color": "Black", "size": "Large"},
        {"name": "Test Chair", "sku": "CH-BLU-S", "color": "Blue", "size": "Small"},
        {"name": "Test Chair", "sku": "CH-BLU-L", "color": "Blue", "size": "Large"},
    ]
    full_matrix = _candidate(
        variants=[
            Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-BLK-S"),
            Variant(options={"Color": "Black", "Size": "Large"}, sku="CH-BLK-L"),
            Variant(options={"Color": "Blue", "Size": "Small"}, sku="CH-BLU-S"),
            Variant(options={"Color": "Blue", "Size": "Large"}, sku="CH-BLU-L"),
        ]
    )
    assert "variant.ungrounded" not in codes(validate_candidate(full_matrix, evidence))


def test_rejects_invalid_taxonomy(evidence):
    """A category outside the Google Product Taxonomy must be rejected.

    Uses a near-miss rather than nonsense, because near-misses are the realistic
    failure: the model produces a plausible path that is not a verbatim entry.
    """
    issues = validate_candidate(
        _candidate(category="Furniture > Chairs > Ergonomic Desk Chairs"), evidence
    )
    assert "category.not_in_taxonomy" in codes(issues)


def test_rejects_unsupported_image(evidence):
    """A variant image URL not present in the harvest must be rejected.

    The fixture URL is one character different from a real one, modelling the way a
    model corrupts a long CDN path rather than inventing a wholly fictional domain.
    """
    tampered = _candidate(
        variants=[
            Variant(
                options={"Color": "Black", "Size": "Small"},
                sku="CH-BLK-S",
                image_urls=["https://cdn.test/img/a1.jpg"],  # real URL is a.jpg
            )
        ]
    )
    assert "image.unsupported" in codes(validate_candidate(tampered, evidence))


def test_rejects_unsupported_identifier(evidence):
    """A SKU that appears nowhere in the evidence must be rejected."""
    invented = _candidate(
        variants=[Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-XXX-9")]
    )
    assert "identifier.unsupported" in codes(validate_candidate(invented, evidence))


def test_deduplicates_variants(evidence):
    """Two variants with identical options are the same configuration listed twice."""
    duplicated = _candidate(
        variants=[
            Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-BLK-S"),
            Variant(options={"Color": "Black", "Size": "Small"}, sku="CH-BLK-S"),
        ]
    )
    issues = validate_candidate(duplicated, evidence)
    assert "variant.duplicate" in codes(issues)
    # A duplicate is a warning, not an error: the fix is to collapse it, not to fail.
    assert all(i.severity == "warning" for i in issues if i.code == "variant.duplicate")


def test_handles_product_without_variants(evidence):
    """A product with no variants is normal, not an error.

    Many PDPs sell exactly one configuration. An empty variant list must pass cleanly,
    otherwise the pipeline would fail on a large share of real pages.
    """
    issues = validate_candidate(_candidate(variants=[]), evidence)
    assert not [i for i in issues if i.severity == "error"]


def test_preserves_variant_specific_price(evidence):
    """A variant priced differently from the product must be accepted when evidenced.

    Per-variant pricing is real (a larger size costs more), so the validator must not
    assume a variant price matching the product price. Here the variant price appears
    in its own evidence record, which is what grounds it.
    """
    evidence.json_subtrees.append(
        {"name": "Test Chair", "sku": "CH-BLU-L", "color": "Blue", "size": "Large", "price": 299.0}
    )
    priced = _candidate(
        variants=[
            Variant(
                options={"Color": "Blue", "Size": "Large"},
                sku="CH-BLU-L",
                price=Price(price=299.0, currency="USD"),
            )
        ]
    )
    issues = validate_candidate(priced, evidence)
    assert not [i for i in issues if i.severity == "error"]


def test_variant_with_no_commercial_data_is_still_valid(evidence):
    """A declared configuration with no price or SKU must survive validation.

    Some PDPs show a colour swatch strip without exposing per-colour pricing anywhere.
    Dropping those would silently destroy variant recall, so nullable commercial fields
    are a supported state - provided the configuration itself is evidenced.
    """
    declared_only = _candidate(
        variants=[Variant(options={"Color": "Black", "Size": "Small"})]
    )
    issues = validate_candidate(declared_only, evidence)
    assert not [i for i in issues if i.severity == "error"]


def test_does_not_select_recommendation_as_primary_product():
    """Content from a recommendation strip must not reach the evidence in the first place.

    This failure class is handled structurally rather than by the validator: once a
    candidate exists, its name and price are just strings with nothing generic left to
    check them against. So the assertion belongs against the extractor - a cross-sell
    block is a high-link-density region and must be suppressed before it can be read.
    """
    import extract

    html = """
    <html><head>
      <meta property="og:title" content="Main Product">
      <link rel="canonical" href="https://example.test/p/main">
    </head><body>
      <div id="main"><h1>Main Product</h1><p>The real description.</p>
        <span>$100.00</span></div>
      <div id="recs">
        <a href="/p/other-1">Other Product One $11.00</a>
        <a href="/p/other-2">Other Product Two $22.00</a>
        <a href="/p/other-3">Other Product Three $33.00</a>
      </div>
    </body></html>
    """
    evidence = extract.run(html)
    blob = "\n".join(evidence.text_blocks)

    assert "The real description." in blob
    for intruder in ("Other Product One", "$11.00", "$22.00", "$33.00"):
        assert intruder not in blob, f"recommendation content leaked into evidence: {intruder}"


def test_rejects_incoherent_price(evidence):
    """A 'was' price below the current price is incoherent regardless of the page."""
    issues = validate_candidate(
        _candidate(price=Price(price=249.0, currency="USD", compare_at_price=99.0)),
        evidence,
    )
    assert "price.insane" in codes(issues)


def test_unsupported_price_is_only_a_warning(evidence):
    """An unverifiable price annotates the result but must not fail it.

    Price formatting varies too much across pages for containment checking to be a
    reliable gate, so this is a drift metric rather than a rejection.
    """
    issues = validate_candidate(
        _candidate(price=Price(price=13579.0, currency="USD")), evidence
    )
    unsupported = [i for i in issues if i.code == "price.unsupported"]
    assert unsupported and all(i.severity == "warning" for i in unsupported)
