"""
pipeline/scraper.py

Headless Chromium scraper using Playwright.
Captures a full-page screenshot plus clean text, headline, and CTA
from any publicly accessible URL.
"""

# stdlib
import re

# third-party
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def scrape_page(url: str) -> dict:
    """
    Scrape a webpage and return its screenshot, HTML, and key text elements.

    Args:
        url: The fully-qualified URL to scrape (must include http/https).

    Returns:
        dict with keys:
            screenshot (bytes): Full-page PNG bytes.
            html (str): Raw page HTML.
            text (str): Clean visible text, truncated to 2000 chars.
            headline (str): Text of the first <h1>, or empty string.
            cta_text (str): Text of the first button/CTA link, or empty string.
            url (str): The original URL passed in.

    Raises:
        ValueError: If the page fails to load or the screenshot is empty.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            response = page.goto(url, wait_until="networkidle", timeout=30_000)
            if response is None or not response.ok:
                status = response.status if response else "no response"
                raise ValueError(f"Page returned non-OK status: {status}")

            # Let lazy-loaded content settle
            page.wait_for_timeout(2000)

            screenshot_bytes: bytes = page.screenshot(full_page=True)
            if not screenshot_bytes:
                raise ValueError("Screenshot returned empty bytes.")

            html: str = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags before text extraction
        for tag in soup.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        text: str = soup.get_text(separator=" ", strip=True)[:2000]

        # Headline
        h1 = soup.find("h1")
        headline: str = h1.get_text(strip=True) if h1 else ""

        # CTA — prefer button, fall back to anchor whose class looks like a CTA
        cta_text: str = ""
        button = soup.find("button")
        if button:
            cta_text = button.get_text(strip=True)
        else:
            cta_pattern = re.compile(r"\bcta\b|\bbtn\b|\bbutton\b", re.IGNORECASE)
            for a in soup.find_all("a"):
                classes = " ".join(a.get("class", []))
                if cta_pattern.search(classes):
                    cta_text = a.get_text(strip=True)
                    break

        return {
            "screenshot": screenshot_bytes,
            "html": html,
            "text": text,
            "headline": headline,
            "cta_text": cta_text,
            "url": url,
        }

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to scrape {url!r}: {exc}") from exc
