"""
Retrieve support articles from support.stackct.com via the Zendesk Help Center API
and write them to gen/training.txt.

Usage:
    python phase_1_training/retrieve_data.py
"""

import html
import json
import os
import re
import time
import urllib.request

BASE_URL = "https://support.stackct.com/api/v2/help_center/en-us"
OUTPUT_FILE = "gen/training.txt"
PER_PAGE = 100
DELAY = 0.5  # seconds between requests — be polite


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_all_articles() -> list[dict]:
    articles = []
    page = 1
    while True:
        url = f"{BASE_URL}/articles.json?per_page={PER_PAGE}&page={page}"
        print(f"  Fetching page {page}...")
        data = fetch_json(url)
        articles.extend(data["articles"])
        if not data.get("next_page"):
            break
        page += 1
        time.sleep(DELAY)
    return articles


def format_article(article: dict) -> str:
    title = article.get("title", "").strip()
    body = strip_html(article.get("body") or "")
    if not body:
        return ""
    return f"# {title}\n\n{body}\n"


def main():
    os.makedirs("gen", exist_ok=True)

    print(f"Fetching articles from {BASE_URL}...")
    articles = fetch_all_articles()
    print(f"Retrieved {len(articles)} articles.")

    sections = []
    for article in articles:
        text = format_article(article)
        if text:
            sections.append(text)

    training_text = "\n\n---\n\n".join(sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(training_text)

    print(f"Wrote {len(sections)} articles ({len(training_text):,} chars) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
