import json
from openai import OpenAI

client = OpenAI()


def generate_weekly_ai_feedback(ml_input: dict, base_result: dict) -> dict:
    prompt = f"""
You are a supportive student wellness coach.

Write supportive, practical, non-clinical advice in simple English.

Weekly student data:
- Average stress level: {ml_input.get('avg_stress_level')}
- Average study hours: {ml_input.get('avg_study_hours')}
- Dominant confidence: {ml_input.get('dominant_confidence')}
- Total deadlines: {ml_input.get('total_deadlines')}
- Completed check-ins: {ml_input.get('completed_checkins')}
- User summary: {ml_input.get('user_summary', '')}

Current analytics from the app:
- Burnout risk: {base_result.get('burnout_risk')}
- Burnout score: {base_result.get('burnout_score')}
- Stress trend: {base_result.get('stress_trend')}
- Predicted next stress: {base_result.get('predicted_next_stress')}
- Wellness score: {base_result.get('wellness_score')}

Return valid JSON with exactly these keys:
recommendation
focus_areas
encouragement

Rules:
- recommendation must be simple, supportive, and practical
- focus_areas must be a list of exactly 3 short strings
- encouragement must be exactly 1 sentence
- do not diagnose any medical condition
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    text = response.output_text.strip()

    # fallback if model wraps output in markdown
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)