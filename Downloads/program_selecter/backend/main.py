"""
FastAPI backend for the Vlerick Programme Recommendation Tool.
Endpoints:
  POST /recommend  - Upload CV + career goals → get recommendations + email
  GET  /programmes - List all 61 programmes with metadata
  GET  /health     - Health check
"""

import json
import os
import glob
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from parsers import parse_file, parse_linkedin_url
from profiler import extract_profile
from classifier import classify_goals
from recommender import recommend

app = FastAPI(
    title="Vlerick Programme Recommender",
    description="AI-powered programme recommendations based on your CV and career goals",
    version="1.0.0",
)

# Allow CORS for the Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROGRAMME_PAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "programme_pages")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/programmes")
def list_programmes():
    """Return metadata for all 61 programmes."""
    programmes = []
    json_files = glob.glob(
        os.path.join(PROGRAMME_PAGES_DIR, "**", "*.json"), recursive=True
    )

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            continue
        programmes.append({
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "category": data.get("url", "").split("/programmes/programmes-in-")[-1].split("/")[0].replace("-", " ").title()
                if "/programmes/programmes-in-" in data.get("url", "") else "",
            "fee": data.get("key_facts", {}).get("fee", ""),
            "format": data.get("key_facts", {}).get("format", ""),
            "location": data.get("key_facts", {}).get("location", ""),
            "description": data.get("description", "")[:200],
        })

    return {"programmes": programmes, "count": len(programmes)}


@app.post("/recommend")
async def get_recommendations(
    file: UploadFile = File(None),
    career_goals: str = Form(""),
    linkedin_url: str = Form(""),
):
    """
    Main recommendation endpoint.
    Upload a CV (PDF/DOCX/TXT) and/or provide career goals text.
    Returns top 3 programme recommendations and a personalised email draft.
    """
    # Step 1: Parse input sources
    cv_text = ""

    if file:
        file_bytes = await file.read()
        cv_text = parse_file(file.filename, file_bytes)

    if linkedin_url and linkedin_url.strip():
        linkedin_text = parse_linkedin_url(linkedin_url.strip())
        if linkedin_text:
            cv_text += "\n\n" + linkedin_text

    if not cv_text and not career_goals:
        return {
            "error": "Please upload a CV or provide career goals.",
            "recommendations": [],
            "email_draft": "",
        }

    # Step 2: Extract structured profile
    combined_text = cv_text
    if career_goals:
        combined_text += f"\n\nCareer Goals: {career_goals}"

    profile = extract_profile(cv_text, career_goals)

    # Step 3: Zero-shot classify career interests
    classify_input = career_goals or profile.get("career_goals", "")
    if profile.get("skills"):
        classify_input += " " + " ".join(profile["skills"])
    if profile.get("industry"):
        classify_input += " " + profile["industry"]

    categories = classify_goals(classify_input, top_k=3)

    # Step 4: RAG search + LLM synthesis
    result = recommend(profile, categories)

    return {
        "profile": profile,
        "top_categories": categories,
        "recommendations": result.get("recommendations", []),
        "email_draft": result.get("email_draft", ""),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
