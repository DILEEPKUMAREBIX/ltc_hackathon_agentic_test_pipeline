"""
UI Inspector — captures real page structure from the live app using Playwright.

Visits each URL, extracts the accessibility tree and all form elements
(inputs, buttons, links) so Agent 3 can write tests against real selectors
instead of guessing them from source code.
"""

import json
from typing import Optional


def _flatten_snapshot(node: dict, depth: int = 0) -> list[str]:
    """Flatten an accessibility snapshot into readable lines."""
    if not node:
        return []
    lines = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    if role and name:
        indent = "  " * depth
        line = f"{indent}[{role}] {name}"
        if value:
            line += f' = "{value}"'
        lines.append(line)

    for child in node.get("children", []):
        lines.extend(_flatten_snapshot(child, depth + 1))
    return lines


def inspect_pages(
    urls: list[str],
    headless: bool = True,
    timeout_ms: int = 10_000,
) -> str:
    """
    Visit each URL with a headless Chromium browser and return a structured
    string describing the page's form elements and accessibility tree.
    Gracefully returns an error message if a page is unreachable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "(playwright not installed — run: pip install playwright && playwright install chromium)"

    sections = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context()
        page = ctx.new_page()

        for url in urls:
            print(f"[ui_inspector] Inspecting {url}")
            try:
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)  # let React render

                # 1. Accessibility snapshot (roles + names)
                snapshot = page.accessibility.snapshot(interesting_only=True)
                tree_lines = _flatten_snapshot(snapshot) if snapshot else ["(empty)"]

                # 2. All interactive elements with real attributes
                elements = page.evaluate("""() => {
                    const sel = 'input, textarea, select, button, a[href], [role="button"]';
                    return [...document.querySelectorAll(sel)].map(el => ({
                        tag:         el.tagName.toLowerCase(),
                        type:        el.type || '',
                        id:          el.id || '',
                        name:        el.name || '',
                        placeholder: el.placeholder || '',
                        ariaLabel:   el.getAttribute('aria-label') || '',
                        label:       (document.querySelector('label[for="' + el.id + '"]')
                                        ?.textContent?.trim()) || '',
                        text:        el.textContent?.trim().slice(0, 80) || '',
                        href:        el.href || '',
                        className:   el.className?.toString().split(' ').slice(0,3).join(' ') || '',
                    }));
                }""")

                # 3. Page title + headings for context
                title = page.title()
                headings = page.evaluate("""() =>
                    [...document.querySelectorAll('h1,h2,h3')].map(h => h.textContent.trim())
                """)

                sections.append(f"""
=== PAGE: {url} ===
Title: {title}
Headings: {headings}

--- Accessibility Tree ---
{chr(10).join(tree_lines[:80])}

--- Interactive Elements ---
{json.dumps(elements[:40], indent=2)}
""")

            except Exception as e:
                sections.append(f"\n=== PAGE: {url} ===\nERROR: {e}\n(Is the app running?)\n")

        browser.close()

    return "\n".join(sections)
