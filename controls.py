"""Reads option groups out of the rendered page.

We do not decide whether a group is a product option. No structural test tells
"Colour" apart from "Sort by" without picking a language. `pipeline` decides that.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

import images
from models import OptionGroup, OptionValue

# Structural bounds, not discrimination thresholds. 
MAX_GROUPS = 24
MAX_VALUES_PER_GROUP = 300
LABEL_HOPS = 4  # how far out to look for a heading that names an unlabelled group

# `background-image: url(...)`. Swatches are routinely painted as CSS backgrounds
# rather than <img>, which is a rendering choice with no bearing on whether the URL is
# a photograph of the product.
_CSS_URL = re.compile(r"url\(\s*['\"]?(?P<url>[^'\")]+)['\"]?\s*\)", re.I)

# Attributes carrying a lazy-loaded image source. 
_LAZY_SRC_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-image")

# Values that are placeholders rather than choices: the empty first entry of a
# `<select>`. 
_PLACEHOLDER_VALUES = {"", "0", "-1", "none", "null", "undefined", "choose", "select"}


# --------------------------------------------------------------------------------
# Accessible naming
# --------------------------------------------------------------------------------


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.text(deep=True, separator=" ")).strip() if node else ""


def _labelled_by(node, root) -> str:
    #Follow `aria-labelledby` to the text of the elements it points at.
    ref = node.attributes.get("aria-labelledby")
    if not ref:
        return ""
    parts = []
    for token in ref.split():
        target = root.css_first(f'[id="{token}"]')
        if target is not None:
            parts.append(_text(target))
    return " ".join(part for part in parts if part).strip()


def _accessible_name(node, root) -> str:
    #The accessible name of an element, in the order ARIA defines.
    attrs = node.attributes
    for candidate in (attrs.get("aria-label"), _labelled_by(node, root), attrs.get("title")):
        if candidate and candidate.strip():
            return candidate.strip()

    image = node.css_first("img")
    if image is not None:
        for key in ("alt", "aria-label", "title"):
            value = image.attributes.get(key)
            if value and value.strip():
                return value.strip()

    return _first_field(_leading_text(node) or _text(node))


def _first_field(name: str) -> str:
    #The first comma separated part of a built up accessible name.

    head = name.split(",", 1)[0].strip()
    return head or name.strip()


def _leading_text(node) -> str:
    #The first run of text inside an element, in document order.

    for descendant in node.iter(include_text=True):
        if descendant.tag == "-text":
            text = re.sub(r"\s+", " ", descendant.text(deep=False) or "").strip()
            if text:
                return text
        else:
            nested = _leading_text(descendant)
            if nested:
                return nested
    return ""


def _group_name(node, root) -> str | None:
    #Name a group of choices, falling back to the heading just before it.
    for source in (node.attributes.get("aria-label"), _labelled_by(node, root)):
        if source and source.strip():
            return source.strip()

    current = node
    for _ in range(LABEL_HOPS):
        sibling = current.prev
        while sibling is not None:
            text = _text(sibling)
            if text:
                return text.split(":")[0].strip() or None
            sibling = sibling.prev
        current = current.parent
        if current is None:
            break
    return None


# --------------------------------------------------------------------------------
# Per-value detail
# --------------------------------------------------------------------------------


def _swatch_url(node, base_url: str | None) -> str | None:
    #The image a choice shows, either an `<img>` or a CSS background.
    for element in [node] + node.css("*"):
        style = element.attributes.get("style") or ""
        if match := _CSS_URL.search(style):
            return _absolutise(match.group("url"), base_url)

    image = node.css_first("img")
    if image is not None:
        for key in _LAZY_SRC_ATTRS:
            value = image.attributes.get(key)
            if value and not value.startswith("data:"):
                return _absolutise(value, base_url)
    return None


def _value_url(node, base_url: str | None) -> str | None:
    #Where a choice links to, when it is a link instead of a control.

    anchor = node.css_first("a[href]")
    if anchor is None:
        current, hops = node.parent, 0
        while current is not None and hops < 3:
            if current.tag == "a" and current.attributes.get("href"):
                anchor = current
                break
            current, hops = current.parent, hops + 1
    if anchor is None:
        return None
    href = anchor.attributes.get("href")
    return _absolutise(href, base_url) if href else None


def _absolutise(url: str | None, base_url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    if not url or url.startswith("data:"):
        return None
    return urljoin(base_url, url) if base_url else url


def _is_selected(node) -> bool:
    #Whether this choice is the one currently selected.
    attrs = node.attributes
    if attrs.get("aria-checked") == "true" or attrs.get("aria-selected") == "true":
        return True
    return "checked" in attrs or "selected" in attrs


def _is_available(node) -> bool | None:
    #Whether a choice is in stock. None when the page does not say.

    attrs = node.attributes
    if attrs.get("aria-disabled") == "true" or "disabled" in attrs:
        return False
    return None


def _title_depth(node) -> int | None:
    #How far this sits from the product title's ancestors, if it is inside them.
    current, hops = node, 0
    while current is not None and hops < 25:
        depth = current.attributes.get(images.TITLE_ATTR)
        if depth is not None:
            try:
                return int(depth)
            except ValueError:
                return None
        current = current.parent
        hops += 1
    return None


# --------------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------------


def _build_values(nodes, root, base_url: str | None) -> list[OptionValue]:
    values: list[OptionValue] = []
    seen: set[str] = set()
    for node in nodes[:MAX_VALUES_PER_GROUP]:
        label = _accessible_name(node, root)
        if not label:
            continue
        label = label.strip()
        key = label.casefold()
        if key in seen or key in _PLACEHOLDER_VALUES:
            continue
        seen.add(key)
        values.append(
            OptionValue(
                label=label,
                selected=_is_selected(node),
                image_url=_swatch_url(node, base_url),
                url=_value_url(node, base_url),
                available=_is_available(node),
            )
        )
    return values


def harvest(tree, base_url: str | None = None) -> list[OptionGroup]:
    #Every group of choices the page renders, in document order.

    groups: list[OptionGroup] = []
    seen_signatures: set[tuple] = set()

    def add(container, value_nodes, name: str | None, declared: bool = False) -> None:
        if len(groups) >= MAX_GROUPS:
            return
        # A region is marked as a recommendation strip when most of its text is link
        # text. 
        if not declared and images.in_skipped_region(container):
            return
        values = _build_values(value_nodes, tree, base_url)
        if not values:
            return
        signature = ((name or "").casefold(), tuple(value.label.casefold() for value in values))
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        groups.append(
            OptionGroup(name=name, values=values, title_depth=_title_depth(container))
        )

    for container in tree.css('[role="radiogroup"]'):
        add(container, container.css('[role="radio"]'), _group_name(container, tree), True)

    for container in tree.css('[role="listbox"]'):
        add(container, container.css('[role="option"]'), _group_name(container, tree), True)

    for container in tree.css("select"):
        name = _group_name(container, tree) or container.attributes.get("name")
        options = container.css("option")

        if options and "disabled" in options[0].attributes:
            options = options[1:]
        add(container, options, name)

    for container in tree.css("fieldset"):
        radios = container.css('input[type="radio"], input[type="checkbox"]')
        if not radios:
            continue
        legend = container.css_first("legend")
        name = _text(legend) or _group_name(container, tree)

        labelled = [_label_for(radio, container) or radio for radio in radios]
        add(container, labelled, name)

    return groups


def _label_for(node, scope):
    #The `<label>` tied to a form control, by `for=` or by wrapping it."
    node_id = node.attributes.get("id")
    if node_id:
        label = scope.css_first(f'label[for="{node_id}"]')
        if label is not None:
            return label
    current, hops = node.parent, 0
    while current is not None and hops < 3:
        if current.tag == "label":
            return current
        current, hops = current.parent, hops + 1
    return None
