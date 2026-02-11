"""
Zero-shot classification of candidate career goals into Vlerick programme categories.
Uses facebook/bart-large-mnli from HuggingFace transformers.
This runs locally — no API key needed.
"""

from transformers import pipeline

# The 12 Vlerick executive education categories
VLERICK_CATEGORIES = [
    "Accounting & Finance",
    "Digital Transformation and AI",
    "Entrepreneurship",
    "General Management",
    "Healthcare Management",
    "Human Resource Management",
    "Innovation Management",
    "Marketing & Sales",
    "Operations & Supply Chain Management",
    "People Management & Leadership",
    "Strategy",
    "Sustainability",
]

# Lazy-loaded classifier (downloads ~1.6GB model on first use, then cached)
_classifier = None


def get_classifier():
    """Get or initialize the zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        print("Loading zero-shot classification model (first time may take a minute)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def classify_goals(text: str, top_k: int = 3) -> list[dict]:
    """
    Classify career goals/profile text against the 12 Vlerick categories.

    Args:
        text: The candidate's career goals, skills, and background combined.
        top_k: Number of top categories to return.

    Returns:
        List of dicts with 'category' and 'score', sorted by score descending.
        Example: [{"category": "Strategy", "score": 0.85}, ...]
    """
    classifier = get_classifier()

    result = classifier(
        text,
        candidate_labels=VLERICK_CATEGORIES,
        multi_label=True,
    )

    # Pair labels with scores and sort
    categories = []
    for label, score in zip(result["labels"], result["scores"]):
        categories.append({"category": label, "score": round(score, 4)})

    # Already sorted by score from the pipeline, just take top_k
    return categories[:top_k]
