"""
RAG-based programme recommender + personalised email generator.
Combines ChromaDB vector search with GPT-4o-mini synthesis.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "vlerick_programmes"

# Lazy-loaded resources
_collection = None


def get_collection():
    """Get or initialize the ChromaDB collection."""
    global _collection
    if _collection is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    return _collection


def search_programmes(query: str, n_results: int = 10) -> list[dict]:
    """
    Search the vector store for programmes matching the query.
    Returns deduplicated programme results with metadata.
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Deduplicate by programme title
    seen_titles = set()
    programmes = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        title = meta.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            programmes.append({
                "title": title,
                "url": meta.get("url", ""),
                "category": meta.get("category", ""),
                "fee": meta.get("fee", ""),
                "format": meta.get("format", ""),
                "location": meta.get("location", ""),
                "start_date": meta.get("start_date", ""),
                "relevance_score": round(1 - dist, 4),
                "description_snippet": doc[:500],
            })

    return programmes


RECOMMEND_PROMPT = """You are an admissions consultant at Vlerick Business School.

Given this candidate's profile and a list of matching programmes, select the TOP 3 best programmes for this person. For each, explain in 2-3 sentences WHY it fits their background and goals.

Then write a warm, personalised outreach email (3-4 paragraphs) that:
- Addresses the candidate by name
- References their specific background and goals
- Introduces the top 3 recommended programmes with brief reasons
- Includes a call to action to learn more or book a consultation

Return ONLY valid JSON:
{
  "recommendations": [
    {
      "title": "Programme Name",
      "url": "https://...",
      "category": "Category",
      "fee": "€X,XXX",
      "format": "X days",
      "location": "City",
      "reason": "Why this programme fits the candidate..."
    }
  ],
  "email_draft": "Full email text here..."
}"""


def recommend(profile: dict, categories: list[dict]) -> dict:
    """
    Generate programme recommendations and a personalised email.

    Args:
        profile: Structured candidate profile from profiler.py
        categories: Top zero-shot categories from classifier.py

    Returns:
        Dict with 'recommendations' list and 'email_draft' string.
    """
    # Build search query from profile + top categories
    query_parts = []
    if profile.get("career_goals"):
        query_parts.append(profile["career_goals"])
    if profile.get("current_role"):
        query_parts.append(f"Current role: {profile['current_role']}")
    if profile.get("industry"):
        query_parts.append(f"Industry: {profile['industry']}")
    if profile.get("skills"):
        query_parts.append(f"Skills: {', '.join(profile['skills'][:5])}")
    for cat in categories[:3]:
        query_parts.append(cat["category"])

    search_query = " | ".join(query_parts)

    # RAG search
    matches = search_programmes(search_query, n_results=15)

    if not matches:
        return {
            "recommendations": [],
            "email_draft": "No matching programmes found.",
        }

    # Build context for LLM
    candidate_summary = json.dumps(profile, indent=2)
    category_summary = ", ".join(
        f"{c['category']} ({c['score']:.0%})" for c in categories[:3]
    )
    programme_list = "\n\n".join(
        f"- {m['title']} ({m['category']})\n"
        f"  Fee: {m['fee']} | Format: {m['format']} | Location: {m['location']}\n"
        f"  URL: {m['url']}\n"
        f"  {m['description_snippet'][:300]}"
        for m in matches[:8]
    )

    user_message = (
        f"CANDIDATE PROFILE:\n{candidate_summary}\n\n"
        f"TOP INTEREST AREAS: {category_summary}\n\n"
        f"MATCHING PROGRAMMES:\n{programme_list}"
    )

    # LLM synthesis
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RECOMMEND_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON response
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return raw matches without LLM synthesis
        result = {
            "recommendations": [
                {
                    "title": m["title"],
                    "url": m["url"],
                    "category": m["category"],
                    "fee": m["fee"],
                    "format": m["format"],
                    "location": m["location"],
                    "reason": f"Matched based on your interest in {m['category']}.",
                }
                for m in matches[:3]
            ],
            "email_draft": content,
        }

    return result
