"""Fetches real headlines from Formula1.com's official RSS feed. Headlines,
links, and dates are displayed verbatim from the feed -- never summarized,
paraphrased, or fabricated. If the feed can't be reached, that's reported
plainly rather than filled in with invented content.
"""

import xml.etree.ElementTree as ET

import requests

F1_RSS_URL = "https://www.formula1.com/en/latest/all.xml"


def fetch_latest_news(team_or_driver: str | None = None, limit: int = 8) -> list[dict]:
    """Returns real news items from the official F1.com feed: [{title, link,
    description, pub_date}, ...]. If `team_or_driver` is given, filters to
    items whose title/description mention it (case-insensitive substring
    match) -- since the feed has no structured team tagging, this is a
    simple keyword filter, not a guarantee of exhaustive coverage."""
    try:
        resp = requests.get(F1_RSS_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return [{"error": f"Could not reach the official F1.com news feed: {e}"}]

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        return [{"error": f"Could not parse the F1.com news feed: {e}"}]

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        if team_or_driver:
            haystack = f"{title} {description}".lower()
            if team_or_driver.lower() not in haystack:
                continue

        items.append({"title": title, "link": link, "description": description, "pub_date": pub_date})
        if len(items) >= limit:
            break

    return items


if __name__ == "__main__":
    print("=== Latest F1 news (unfiltered) ===")
    for item in fetch_latest_news(limit=5):
        print(f"- {item.get('title')}")
        print(f"  {item.get('link')}")

    print("\n=== Filtered: 'Ferrari' ===")
    for item in fetch_latest_news("Ferrari", limit=5):
        print(f"- {item.get('title')}")
