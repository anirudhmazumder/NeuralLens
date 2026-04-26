"""
tests/test_scraper.py

Standalone test for pipeline/scraper.py.
Run with:  python tests/test_scraper.py
"""

# stdlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party
from dotenv import load_dotenv

load_dotenv()

# local
from pipeline.scraper import scrape_page

TEST_URL = "https://example.com"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")


def test_scrape():
    print(f"\nTesting scrape_page({TEST_URL!r}) ...")
    try:
        result = scrape_page(TEST_URL)

        assert isinstance(result["screenshot"], bytes), "screenshot must be bytes"
        assert len(result["screenshot"]) > 0, "screenshot must not be empty"
        assert isinstance(result["html"], str), "html must be str"
        assert isinstance(result["text"], str), "text must be str"
        assert len(result["text"]) <= 2000, "text must be truncated to 2000 chars"
        assert result["url"] == TEST_URL, "url must match input"

        print(f"  headline   : {result['headline']!r}")
        print(f"  cta_text   : {result['cta_text']!r}")
        print(f"  text length: {len(result['text'])} chars")
        print(f"  screenshot : {len(result['screenshot'])} bytes")

        # Save screenshot for manual inspection
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "screenshot.png")
        with open(out_path, "wb") as f:
            f.write(result["screenshot"])
        print(f"  saved to   : {out_path}")

        print("\nPASS: scrape_page")
        return result

    except Exception as exc:
        print(f"\nFAIL: scrape_page — {exc}")
        raise


if __name__ == "__main__":
    test_scrape()
