"""
pipeline/optimization_loop.py

Orchestrates the full NeuralLens analysis pipeline end-to-end:
  scrape → TRIBE v2 → ROI extraction → NES scoring → heatmap →
  Cloudinary upload → Gemma suggestions → apply changes →
  re-screenshot → re-score → upload after images.
"""

# stdlib
import time
import traceback

# third-party
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# local
from pipeline.scraper import scrape_page
from pipeline.tribe_inference import run_tribe_inference, activations_to_heatmap
from pipeline.roi_extractor import extract_roi_values
from pipeline.nes_scorer import compute_nes
from pipeline.gemma_optimizer import get_suggestions
from integrations.cloudinary_client import upload_image, create_overlay


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def apply_changes_to_html(html: str, changes: list) -> str:
    """
    Apply a list of Gemma-suggested changes to raw HTML using BeautifulSoup.

    Args:
        html: Full page HTML string.
        changes: List of change dicts, each with keys:
                 selector, property, old_value, new_value.

    Returns:
        Modified HTML string.  Elements not found by their selector are
        silently skipped.
    """
    soup = BeautifulSoup(html, "html.parser")

    for change in changes:
        selector = change.get("selector", "")
        prop = change.get("property", "")
        new_value = change.get("new_value", "")

        element = soup.select_one(selector)
        if element is None:
            continue

        if prop == "innerHTML":
            element.string = new_value
        elif prop.startswith("style."):
            css_prop = prop[len("style."):]
            existing_style = element.get("style", "")
            # Append / override the single CSS property
            element["style"] = f"{existing_style}; {css_prop}: {new_value}".strip("; ")

    return str(soup)


def screenshot_html(html: str) -> bytes:
    """
    Render an HTML string in a headless Chromium browser and take a screenshot.

    Args:
        html: Full HTML to render (may be a modified version of the original).

    Returns:
        Full-page PNG as bytes.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_content(html, wait_until="networkidle")
        screenshot_bytes: bytes = page.screenshot(full_page=True)
        browser.close()
    return screenshot_bytes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(url: str, industry: str) -> dict:
    """
    Execute the complete NeuralLens analysis and return structured results.

    Args:
        url: Publicly accessible page URL.
        industry: Business vertical (e.g. "e-commerce", "SaaS", "restaurant").

    Returns:
        dict containing NES scores (before/after), delta, ROI values,
        Cloudinary image URLs, Gemma change suggestions, and metadata.

    Raises:
        RuntimeError: Wraps any step-level exception with contextual info.
    """
    ts = int(time.time())

    # ------------------------------------------------------------------
    # Step 1: Scrape
    # ------------------------------------------------------------------
    try:
        print("Step 1/9: Scraping page...")
        scraped = scrape_page(url)
        screenshot_before = scraped["screenshot"]
        html = scraped["html"]
        text = scraped["text"]
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 1 (scrape) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 2: TRIBE v2 — before
    # ------------------------------------------------------------------
    try:
        print("Step 2/9: Running TRIBE v2...")
        activations_before = run_tribe_inference(screenshot_before, text)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 2 (TRIBE v2) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 3: ROI extraction — before
    # ------------------------------------------------------------------
    try:
        print("Step 3/9: Extracting brain regions...")
        roi_before = extract_roi_values(activations_before)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 3 (ROI extraction) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 4: NES scoring — before
    # ------------------------------------------------------------------
    try:
        print("Step 4/9: Computing NES score...")
        nes_before_result = compute_nes(roi_before)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 4 (NES scoring) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 5: Heatmap — before
    # ------------------------------------------------------------------
    try:
        print("Step 5/9: Generating heatmap...")
        heatmap_before = activations_to_heatmap(activations_before, screenshot_before)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 5 (heatmap generation) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 6: Upload to Cloudinary — before images
    # ------------------------------------------------------------------
    try:
        print("Step 6/9: Uploading to Cloudinary...")
        before_screenshot_id = f"neurallens_{ts}_before_screenshot"
        before_heatmap_id = f"neurallens_{ts}_before_heatmap"

        screenshot_before_url = upload_image(screenshot_before, before_screenshot_id)
        upload_image(heatmap_before, before_heatmap_id)
        overlay_before_url = create_overlay(before_screenshot_id, before_heatmap_id)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 6 (Cloudinary upload) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 7: Gemma suggestions
    # ------------------------------------------------------------------
    try:
        print("Step 7/9: Getting Gemma suggestions...")
        suggestions = get_suggestions(html, nes_before_result, industry)
        changes = suggestions.get("changes", [])
        summary = suggestions.get("summary", "")
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 7 (Gemma optimisation) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 8: Apply changes and re-screenshot
    # ------------------------------------------------------------------
    try:
        print("Step 8/9: Applying changes...")
        modified_html = apply_changes_to_html(html, changes)
        screenshot_after = screenshot_html(modified_html)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 8 (apply changes) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 9: TRIBE v2, ROI, NES, heatmap — after
    # ------------------------------------------------------------------
    try:
        print("Step 9/9: Re-scoring...")
        activations_after = run_tribe_inference(screenshot_after, text)
        roi_after = extract_roi_values(activations_after)
        nes_after_result = compute_nes(roi_after)
        heatmap_after = activations_to_heatmap(activations_after, screenshot_after)

        after_screenshot_id = f"neurallens_{ts}_after_screenshot"
        after_heatmap_id = f"neurallens_{ts}_after_heatmap"

        screenshot_after_url = upload_image(screenshot_after, after_screenshot_id)
        upload_image(heatmap_after, after_heatmap_id)
        overlay_after_url = create_overlay(after_screenshot_id, after_heatmap_id)
    except Exception as exc:
        traceback.print_exc()
        raise RuntimeError(f"Step 9 (re-scoring) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    nes_before = nes_before_result["nes_total"]
    nes_after = nes_after_result["nes_total"]

    return {
        "nes_before": nes_before,
        "nes_after": nes_after,
        "delta": round(nes_after - nes_before, 1),
        "profile_before": nes_before_result["profile"],
        "profile_after": nes_after_result["profile"],
        "valence_before": nes_before_result["valence"],
        "valence_after": nes_after_result["valence"],
        "arousal_before": nes_before_result["arousal"],
        "arousal_after": nes_after_result["arousal"],
        "roi_before": nes_before_result["roi_values"],
        "roi_after": nes_after_result["roi_values"],
        "overlay_before_url": overlay_before_url,
        "overlay_after_url": overlay_after_url,
        "screenshot_before_url": screenshot_before_url,
        "screenshot_after_url": screenshot_after_url,
        "changes": changes,
        "summary": summary,
        "issues": nes_before_result["issues"],
        "url": url,
        "industry": industry,
    }
