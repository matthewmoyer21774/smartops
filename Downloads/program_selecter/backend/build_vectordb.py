"""
Builds a ChromaDB vector store from the scraped programme JSON files.
Each programme becomes one or more documents with metadata.
Run this once before starting the API server.
"""

import json
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

PROGRAMME_PAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "programme_pages")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "vlerick_programmes"


def load_programmes():
    """Load all programme JSON files from the scraped data."""
    programmes = []
    json_files = glob.glob(os.path.join(PROGRAMME_PAGES_DIR, "**", "*.json"), recursive=True)

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Skip files that had errors during scraping
        if "error" in data:
            continue

        programmes.append(data)

    return programmes


def build_document(prog):
    """
    Build a single text document from a programme's scraped data.
    This combines all relevant fields into a searchable passage.
    """
    parts = []

    title = prog.get("title", "").strip()
    if title:
        parts.append(f"Programme: {title}")

    subtitle = prog.get("subtitle", "").strip()
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")

    # Key facts
    kf = prog.get("key_facts", {})
    if kf:
        facts = []
        for k, v in kf.items():
            facts.append(f"{k}: {v}")
        parts.append("Key Facts: " + " | ".join(facts))

    description = prog.get("description", "").strip()
    if description:
        parts.append(f"Description: {description}")

    # Programme content sections (deduplicated)
    seen_content = set()
    for section in prog.get("sections", []):
        heading = section.get("heading", "").strip()
        content = section.get("content", "").strip()
        if content and content not in seen_content and len(content) > 20:
            seen_content.add(content)
            parts.append(f"{heading}: {content[:2000]}")

    # Foldable sections (often contain module details)
    for fold in prog.get("foldable_sections", []):
        fold = fold.strip()
        if fold and fold not in seen_content and len(fold) > 20:
            seen_content.add(fold)
            parts.append(fold[:2000])

    # Testimonials
    for t in prog.get("testimonials", []):
        t = t.strip()
        if t and len(t) > 20:
            parts.append(f"Testimonial: {t[:500]}")

    return "\n\n".join(parts)


def build_metadata(prog):
    """Extract metadata for a programme document."""
    kf = prog.get("key_facts", {})
    url = prog.get("url", "")

    # Extract category from URL
    category = ""
    if "/programmes/programmes-in-" in url:
        cat_part = url.split("/programmes/programmes-in-")[1].split("/")[0]
        category = cat_part.replace("-", " ").title()

    return {
        "title": prog.get("title", "").strip(),
        "url": url,
        "category": category,
        "fee": kf.get("fee", ""),
        "format": kf.get("format", ""),
        "location": kf.get("location", ""),
        "start_date": kf.get("start_date", ""),
    }


def main():
    print("Loading scraped programme data...")
    programmes = load_programmes()
    print(f"  Found {len(programmes)} programmes\n")

    # Set up ChromaDB with sentence-transformers embeddings
    print("Initializing ChromaDB with sentence-transformers...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (for clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Build and add documents
    documents = []
    metadatas = []
    ids = []

    for i, prog in enumerate(programmes):
        doc = build_document(prog)
        meta = build_metadata(prog)

        if not doc.strip():
            continue

        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"prog_{i}")

        print(f"  [{i+1}/{len(programmes)}] {meta['title'][:50]}... "
              f"({len(doc)} chars, category: {meta['category']})")

    # Add all documents to ChromaDB
    print(f"\nEmbedding {len(documents)} documents...")
    # ChromaDB handles batching internally
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"\n{'=' * 50}")
    print(f"  Vector DB built successfully!")
    print(f"  Documents: {collection.count()}")
    print(f"  Location:  {CHROMA_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
