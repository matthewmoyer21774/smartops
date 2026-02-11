"""
Vlerick Programme URL Harvester
Queries the Algolia search API directly to get ALL Executive Education
programme URLs. The credentials are extracted from Vlerick's public
JavaScript bundle (search-only key).
"""

import json
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.vlerick.com"

# Algolia search-only credentials (public, extracted from JS bundle 1898)
ALGOLIA_APP_ID = "3VV2ZA1L8V"
ALGOLIA_API_KEY = "3e2dcd8477bfc4fc03da7a12517288fd"
ALGOLIA_INDEX = "prd_website_sortingByDate"
ALGOLIA_SEARCH_URL = (
    f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX}/query"
)


def query_algolia(filters="", hits_per_page=200):
    """
    Query the Algolia search API directly.
    Returns all hits matching the filter.
    """
    headers = {
        "X-Algolia-Application-Id": ALGOLIA_APP_ID,
        "X-Algolia-API-Key": ALGOLIA_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "params": f"hitsPerPage={hits_per_page}&filters={filters}"
    }

    resp = requests.post(ALGOLIA_SEARCH_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"   Algolia returned {data.get('nbHits', 0)} total hits, "
          f"got {len(data.get('hits', []))} in this page")

    return data


def harvest_from_algolia():
    """
    Query Algolia for all Executive Education programmes.
    Returns a set of clean programme URLs.
    """
    all_urls = set()
    all_hits = []

    # Query 1: All Executive Education programmes
    print("   Query: Executive Education programmes...")
    data = query_algolia(
        filters='programme_type:"Executive Education"',
        hits_per_page=500
    )
    all_hits.extend(data.get("hits", []))

    # If there are more hits than returned, paginate
    total_hits = data.get("nbHits", 0)
    if total_hits > len(data.get("hits", [])):
        page = 1
        while len(all_hits) < total_hits:
            print(f"   Fetching page {page + 1}...")
            headers = {
                "X-Algolia-Application-Id": ALGOLIA_APP_ID,
                "X-Algolia-API-Key": ALGOLIA_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "params": f"hitsPerPage=500&page={page}"
                          f'&filters=programme_type:"Executive Education"'
            }
            resp = requests.post(
                ALGOLIA_SEARCH_URL, headers=headers, json=payload, timeout=30
            )
            resp.raise_for_status()
            page_data = resp.json()
            hits = page_data.get("hits", [])
            if not hits:
                break
            all_hits.extend(hits)
            page += 1

    # Query 2: Also try without filter to catch any programmes not tagged
    print("   Query: All programmes (unfiltered)...")
    data2 = query_algolia(
        filters='content_type:"programme"',
        hits_per_page=500
    )
    all_hits.extend(data2.get("hits", []))

    # Extract URLs from hits
    for hit in all_hits:
        url = hit.get("url", "")
        if not url:
            continue
        if url.startswith("/"):
            url = BASE_URL + url

        # Only keep actual programme pages
        if "/programmes/programmes-in-" not in url:
            continue

        # Clean URL (remove anchors, query strings)
        clean_url = url.split("#")[0].split("?")[0]

        # Skip brochure sub-pages and category-level pages
        if clean_url.endswith("/brochure/"):
            continue

        # Must have a programme name after the category
        # Pattern: /en/programmes/programmes-in-{category}/{programme-name}/
        path = clean_url.rstrip("/")
        segments = path.split("/")
        # segments: ['https:', '', 'www.vlerick.com', 'en', 'programmes',
        #            'programmes-in-category', 'programme-name']
        if len(segments) >= 7:
            all_urls.add(clean_url)

    return all_urls, all_hits


def fallback_static_harvest():
    """
    Fallback: scrape the static domain pages to catch any programmes
    that might not appear in Algolia results.
    """
    DOMAIN_URLS = [
        "https://www.vlerick.com/en/accounting-finance-programmes-for-professionals/",
        "https://www.vlerick.com/en/digital-transformation-ai-programmes-for-professionals/",
        "https://www.vlerick.com/en/general-management-programmes-for-professionals/",
        "https://www.vlerick.com/en/operations-supply-chain-management-programmes-for-professionals/",
        "https://www.vlerick.com/en/human-resource-management-programmes-for-professionals/",
        "https://www.vlerick.com/en/innovation-management-programmes-for-professionals/",
        "https://www.vlerick.com/en/marketing-sales-programmes-for-professionals/",
        "https://www.vlerick.com/en/people-management-leadership-programmes-for-professionals/",
        "https://www.vlerick.com/en/strategy-programmes-for-professionals/",
        "https://www.vlerick.com/en/sustainability-programmes-for-professionals/",
        "https://www.vlerick.com/en/vlerick-entrepreneurship-academy/",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    urls = set()

    for domain_url in DOMAIN_URLS:
        try:
            resp = requests.get(domain_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/programmes/programmes-in-" in href:
                    if href.startswith("/"):
                        href = BASE_URL + href
                    clean_url = href.split("#")[0].split("?")[0]
                    if not clean_url.endswith("/brochure/"):
                        urls.add(clean_url)
        except Exception as e:
            print(f"  Warning: Failed to fetch {domain_url}: {e}")
        time.sleep(0.5)

    return urls


if __name__ == "__main__":
    print("=" * 60)
    print("  Vlerick Programme URL Harvester")
    print("=" * 60)

    # Step 1: Query Algolia directly for all programmes
    print("\n--- Phase 1: Algolia API (Direct Query) ---")
    algolia_urls, raw_hits = harvest_from_algolia()
    print(f"   > Found {len(algolia_urls)} programme URLs via Algolia.")

    # Step 2: Fallback static scrape
    print("\n--- Phase 2: Static Scrape (Fallback) ---")
    static_urls = fallback_static_harvest()
    print(f"   > Found {len(static_urls)} programme URLs via static scrape.")

    # Merge and deduplicate
    all_urls = algolia_urls | static_urls

    # Final cleanup: remove category-level pages (no programme name)
    clean_urls = set()
    for url in all_urls:
        path = url.rstrip("/")
        segments = path.split("/")
        # Must have: ...programmes-in-{category}/{programme-name}
        # That means at least 7 segments for a full URL
        if len(segments) >= 7:
            # Check last segment isn't a category page
            last = segments[-1]
            if not last.startswith("programmes-in-"):
                clean_urls.add(url)

    sorted_urls = sorted(clean_urls)

    # Save URLs
    with open("vlerick_all_urls.json", "w") as f:
        json.dump(sorted_urls, f, indent=4)

    # Also save full Algolia hit data for the scraper to use later
    # (contains title, description, etc. already!)
    programme_data = []
    seen = set()
    for hit in raw_hits:
        url = hit.get("url", "")
        if url.startswith("/"):
            url = BASE_URL + url
        clean = url.split("#")[0].split("?")[0]
        if clean in clean_urls and clean not in seen:
            seen.add(clean)
            programme_data.append({
                "url": clean,
                "title": hit.get("title", ""),
                "description": hit.get("description", ""),
                "field_of_interest": hit.get("field_of_interest", []),
                "programme_type": hit.get("programme_type", []),
                "format": hit.get("format", []),
                "experience_level": hit.get("experience_level", []),
                "location": hit.get("location", []),
                "start_period": hit.get("start_period", []),
            })

    with open("vlerick_programmes_data.json", "w") as f:
        json.dump(programme_data, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  TOTAL UNIQUE PROGRAMMES: {len(sorted_urls)}")
    print(f"  Saved URLs to:  vlerick_all_urls.json")
    print(f"  Saved data to:  vlerick_programmes_data.json")
    print(f"{'=' * 60}")

    # Print all URLs grouped by category
    from collections import defaultdict
    categories = defaultdict(list)
    for url in sorted_urls:
        parts = url.rstrip("/").split("/")
        cat = parts[5] if len(parts) > 5 else "other"
        cat = cat.replace("programmes-in-", "").replace("-", " ").title()
        categories[cat].append(url)

    for cat in sorted(categories):
        print(f"\n  [{cat}] ({len(categories[cat])} programmes)")
        for url in categories[cat]:
            name = url.rstrip("/").split("/")[-1].replace("-", " ").title()
            print(f"    - {name}")
            print(f"      {url}")
