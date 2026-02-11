"""
Vlerick Programme Page Scraper
Reads vlerick_all_urls.json and scrapes each programme page,
saving full content as individual files + a combined JSON database.
Uses Playwright (headless browser) to render JS-heavy pages.
Saves after each page so no data is lost if interrupted.
"""

import json
import os
import re
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.vlerick.com"
URLS_FILE = "vlerick_all_urls.json"
OUTPUT_DIR = "programme_pages"
COMBINED_FILE = "programmes_database.json"


def slugify(text):
    """Convert text to a safe filename."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def extract_programme_data(page, url):
    """
    Extract all information from a rendered programme page.
    Returns a dict with all scraped data.
    """
    data = {"url": url}

    # --- Title & Subtitle ---
    try:
        h1 = page.locator("h1").first
        data["title"] = h1.inner_text().strip() if h1.count() > 0 else ""
    except Exception:
        data["title"] = ""

    try:
        subtitle = page.locator(".c-hero__subtitle").first
        data["subtitle"] = subtitle.inner_text().strip() if subtitle.count() > 0 else ""
    except Exception:
        data["subtitle"] = ""

    # --- Hero Key Facts (Fee, Format, Location, Date, Duration) ---
    data["key_facts"] = {}
    try:
        # The key facts ribbon items in the hero section
        ribbon_items = page.locator("[class*='ribbon'] [class*='item'], [class*='hero'] [class*='key-fact'], [class*='hero-bottom']")
        if ribbon_items.count() == 0:
            # Try alternative selectors for fact items
            ribbon_items = page.locator(".c-hero__bottom-ribbon-item, .c-keyfact, [data-kontent-element-codename*='key']")

        for i in range(ribbon_items.count()):
            try:
                item_text = ribbon_items.nth(i).inner_text().strip()
                if item_text:
                    lines = [l.strip() for l in item_text.split("\n") if l.strip()]
                    if len(lines) >= 2:
                        data["key_facts"][lines[0]] = " ".join(lines[1:])
            except Exception:
                continue
    except Exception:
        pass

    # --- Try extracting facts from the full page text with known labels ---
    full_text = ""
    try:
        full_text = page.locator("body").inner_text()
    except Exception:
        pass

    fact_patterns = {
        "fee": r"(?:Fee|Price|Cost)[:\s]*([€$£][\d,.]+[^\n]*)",
        "duration": r"(?:Duration)[:\s]*(\d+[^\n]*)",
        "format": r"(?:Format)[:\s]*([^\n]+)",
        "location": r"(?:Location)[:\s]*([^\n]+)",
        "language": r"(?:Language)[:\s]*([^\n]+)",
        "start_date": r"(?:Upcoming edition|Start date|Next edition|Starting)[:\s]*([^\n]+)",
    }
    for key, pattern in fact_patterns.items():
        if key not in data["key_facts"]:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                data["key_facts"][key] = match.group(1).strip()

    # --- Description / Intro ---
    data["description"] = ""
    try:
        intro = page.locator(".c-intro, [class*='intro'], [class*='lead-text']").first
        if intro.count() > 0:
            data["description"] = intro.inner_text().strip()
    except Exception:
        pass

    # --- All body sections ---
    data["sections"] = []
    try:
        # Get all content sections in the body
        body = page.locator("#vlerick\\:body, .c-body, main")
        if body.count() > 0:
            # Find all heading + content pairs
            headings = body.locator("h2, h3")
            for i in range(headings.count()):
                try:
                    heading_text = headings.nth(i).inner_text().strip()
                    if heading_text and len(heading_text) > 1:
                        # Get the next sibling content
                        parent = headings.nth(i).locator("..")
                        section_text = parent.inner_text().strip() if parent.count() > 0 else ""
                        data["sections"].append({
                            "heading": heading_text,
                            "content": section_text
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # --- Foldable/Accordion content (programme structure, modules, etc.) ---
    data["foldable_sections"] = []
    try:
        foldables = page.locator("[class*='foldable'], [class*='accordion'], [class*='collapse'], details")
        for i in range(foldables.count()):
            try:
                # Click to expand if needed
                try:
                    foldables.nth(i).click(timeout=1000)
                    page.wait_for_timeout(300)
                except Exception:
                    pass

                fold_text = foldables.nth(i).inner_text().strip()
                if fold_text and len(fold_text) > 10:
                    data["foldable_sections"].append(fold_text)
            except Exception:
                continue
    except Exception:
        pass

    # --- Testimonials ---
    data["testimonials"] = []
    try:
        testimonials = page.locator("[class*='testimonial'], [class*='quote'], blockquote")
        for i in range(testimonials.count()):
            try:
                t_text = testimonials.nth(i).inner_text().strip()
                if t_text and len(t_text) > 20:
                    data["testimonials"].append(t_text)
            except Exception:
                continue
    except Exception:
        pass

    # --- Contact information ---
    data["contact"] = {}
    try:
        contact_section = page.locator("[class*='contact-card'], [class*='contact']")
        for i in range(min(contact_section.count(), 3)):
            try:
                c_text = contact_section.nth(i).inner_text().strip()
                if c_text and len(c_text) > 10:
                    data["contact"]["info"] = c_text
                    break
            except Exception:
                continue
    except Exception:
        pass

    # --- Full page text (clean version for RAG/embeddings) ---
    data["full_text"] = ""
    try:
        # Get the main content area text, excluding header/footer/nav
        main_content = page.locator("#vlerick\\:body, .c-body")
        if main_content.count() > 0:
            data["full_text"] = main_content.inner_text().strip()
        else:
            # Fallback: get everything between hero and footer
            data["full_text"] = full_text
    except Exception:
        data["full_text"] = full_text

    return data


def save_programme_file(data, output_dir):
    """Save a programme's data as both JSON and text file."""
    # Create filename from URL slug
    url_slug = data["url"].rstrip("/").split("/")[-1]
    category = data["url"].rstrip("/").split("/")[-2].replace("programmes-in-", "")

    # Create category subdirectory
    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(cat_dir, f"{url_slug}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Save readable text file
    txt_path = os.path.join(cat_dir, f"{url_slug}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"PROGRAMME: {data.get('title', 'N/A')}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"URL: {data['url']}\n")
        f.write(f"Subtitle: {data.get('subtitle', '')}\n\n")

        if data.get("key_facts"):
            f.write("--- KEY FACTS ---\n")
            for k, v in data["key_facts"].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if data.get("description"):
            f.write("--- DESCRIPTION ---\n")
            f.write(f"{data['description']}\n\n")

        if data.get("sections"):
            f.write("--- PROGRAMME CONTENT ---\n")
            for section in data["sections"]:
                f.write(f"\n## {section['heading']}\n")
                f.write(f"{section['content']}\n")
            f.write("\n")

        if data.get("foldable_sections"):
            f.write("--- DETAILED SECTIONS ---\n")
            for fold in data["foldable_sections"]:
                f.write(f"\n{fold}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

        if data.get("testimonials"):
            f.write("--- TESTIMONIALS ---\n")
            for t in data["testimonials"]:
                f.write(f"\n{t}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

        if data.get("contact", {}).get("info"):
            f.write("--- CONTACT ---\n")
            f.write(f"{data['contact']['info']}\n\n")

        f.write("--- FULL PAGE TEXT ---\n")
        f.write(data.get("full_text", ""))
        f.write("\n")

    return json_path, txt_path


def load_progress(output_dir):
    """Load already-scraped URLs to allow resuming."""
    scraped = set()
    combined_path = os.path.join(output_dir, COMBINED_FILE)
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            for item in existing:
                scraped.add(item.get("url", ""))
    return scraped


def save_combined(all_data, output_dir):
    """Save the combined database."""
    combined_path = os.path.join(output_dir, COMBINED_FILE)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Load URLs
    with open(URLS_FILE, "r") as f:
        urls = json.load(f)

    print(f"{'=' * 60}")
    print(f"  Vlerick Programme Scraper")
    print(f"  {len(urls)} programmes to scrape")
    print(f"{'=' * 60}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load any previously scraped data (for resume support)
    already_scraped = load_progress(OUTPUT_DIR)
    all_data = []
    combined_path = os.path.join(OUTPUT_DIR, COMBINED_FILE)
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

    remaining = [u for u in urls if u not in already_scraped]
    if already_scraped:
        print(f"  Resuming: {len(already_scraped)} already done, {len(remaining)} remaining\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for i, url in enumerate(remaining, 1):
            prog_name = url.rstrip("/").split("/")[-1]
            print(f"[{i}/{len(remaining)}] Scraping: {prog_name}...")

            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)

                # Scroll down to trigger lazy-loaded content
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)

                # Try to expand all foldable sections
                try:
                    expand_buttons = page.locator(
                        "[class*='foldable'] button, "
                        "[class*='accordion'] button, "
                        "button[aria-expanded='false']"
                    )
                    for j in range(expand_buttons.count()):
                        try:
                            expand_buttons.nth(j).click(timeout=500)
                            page.wait_for_timeout(200)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Extract data
                data = extract_programme_data(page, url)

                # Save individual files
                json_path, txt_path = save_programme_file(data, OUTPUT_DIR)

                # Add to combined and save immediately
                all_data.append(data)
                save_combined(all_data, OUTPUT_DIR)

                print(f"         Saved: {json_path}")

            except Exception as e:
                print(f"         ERROR: {e}")
                # Save error record so we know it failed
                all_data.append({"url": url, "error": str(e)})
                save_combined(all_data, OUTPUT_DIR)

            # Small delay between requests
            time.sleep(1)

        browser.close()

    # Final summary
    success = sum(1 for d in all_data if "error" not in d)
    errors = sum(1 for d in all_data if "error" in d)

    print(f"\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"  Total: {len(all_data)} | Success: {success} | Errors: {errors}")
    print(f"  Files saved to: {OUTPUT_DIR}/")
    print(f"  Combined DB:    {OUTPUT_DIR}/{COMBINED_FILE}")
    print(f"{'=' * 60}")
