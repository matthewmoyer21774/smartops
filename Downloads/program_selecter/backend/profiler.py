"""
LLM-based profile extraction from CV/resume text.
Uses OpenAI GPT-4o-mini to extract structured candidate information.
"""

import json
import os
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert HR analyst. Given a candidate's CV/resume text and their stated career goals, extract a structured profile.

Return ONLY valid JSON with these fields:
{
  "name": "candidate name",
  "current_role": "current job title",
  "years_experience": 0,
  "industry": "primary industry",
  "skills": ["skill1", "skill2", "skill3"],
  "education": "highest education level and field",
  "career_goals": "summarized career aspirations",
  "seniority": "junior|mid|senior|executive"
}

If a field cannot be determined, use null. For skills, list the top 5-8 most relevant professional skills."""


def extract_profile(cv_text: str, career_goals: str = "") -> dict:
    """
    Extract a structured profile from CV text using GPT-4o-mini.
    Returns a dict with candidate profile fields.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    user_message = f"CV/Resume:\n{cv_text[:6000]}"
    if career_goals:
        user_message += f"\n\nStated career goals:\n{career_goals}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        profile = json.loads(content)
    except json.JSONDecodeError:
        profile = {
            "name": None,
            "current_role": None,
            "years_experience": None,
            "industry": None,
            "skills": [],
            "education": None,
            "career_goals": career_goals,
            "seniority": None,
        }

    # Ensure career_goals is populated
    if not profile.get("career_goals") and career_goals:
        profile["career_goals"] = career_goals

    return profile
