"""
AstroTeric — Backend (Ephemeris + Assistant)
=============================================
  1. EPHEMERIS  — true planetary positions, Ascendant, Midheaven.
  2. ASSISTANT  — a grounded, honest chart-and-esoteric assistant.

The assistant calls the Anthropic API directly over HTTPS (httpx)
rather than through the anthropic SDK, which sidesteps SDK/runtime
connection quirks on some hosts.

Endpoints:
  GET  /        — health check
  POST /chart   — compute a natal chart
  POST /ask     — ask the AstroTeric assistant
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import swisseph as swe
import httpx

app = FastAPI(title="AstroTeric Backend", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY,
    "Venus": swe.VENUS, "Mars": swe.MARS, "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN, "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}


def sign_of(longitude: float) -> dict:
    longitude = longitude % 360.0
    index = int(longitude // 30)
    return {
        "sign": SIGNS[index],
        "degree": round(longitude - index * 30, 2),
        "longitude": round(longitude, 4),
    }


# ════════════════════════════════════════════════════════════════
# EPHEMERIS
# ════════════════════════════════════════════════════════════════
class ChartRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int = 12
    minute: int = 0
    tz_offset: float = 0.0
    latitude: float
    longitude: float


@app.get("/")
def health():
    return {
        "status": "alive",
        "service": "AstroTeric Backend",
        "version": "2.1.0",
        "ephemeris": True,
        "assistant": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/chart")
def compute_chart(req: ChartRequest):
    try:
        local = datetime(
            req.year, req.month, req.day, req.hour, req.minute,
            tzinfo=timezone(timedelta(hours=req.tz_offset)),
        )
        ut = local.astimezone(timezone.utc)
        jd = swe.julday(
            ut.year, ut.month, ut.day,
            ut.hour + ut.minute / 60.0 + ut.second / 3600.0,
        )
        planets = {}
        for name, code in PLANETS.items():
            result, _ = swe.calc_ut(jd, code)
            info = sign_of(result[0])
            info["retrograde"] = result[3] < 0
            planets[name] = info
        houses, ascmc = swe.houses(jd, req.latitude, req.longitude, b"P")
        return {
            "ok": True,
            "julian_day": round(jd, 6),
            "planets": planets,
            "ascendant": sign_of(ascmc[0]),
            "midheaven": sign_of(ascmc[1]),
            "houses": [sign_of(h) for h in houses],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chart calculation failed: {e}")


# ════════════════════════════════════════════════════════════════
# ASSISTANT
# ════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are the AstroTeric Assistant — a knowledgeable, honest guide \
within the AstroTeric app, which blends numerology, Western astrology, Chinese \
zodiac, and BaZi (Four Pillars).

YOUR ROLE
You are a practitioner's tool and a guide for serious newcomers. Two kinds of \
questions: (A) questions about a specific chart, when chart data is provided, and \
(B) general questions about esoteric systems and their concepts. Answer both.

YOUR CHARACTER
- Honest above all. Never sugar-coat, never give false hope, never inflate. If a \
placement is challenging, say so plainly and constructively. People come to you \
because they want the real thing, not flattery.
- Grounded. When chart data is provided, base your answer on THAT data. Do not \
invent placements. If asked about something not in the data, say it isn't available.
- Clear. Explain esoteric terms in plain language. A newcomer should understand you; \
a practitioner should respect you.
- Humble about the nature of this knowledge. These are interpretive traditions, not \
deterministic science. Frame readings as lenses for reflection, to be tested against \
real life — never as fixed fate or prediction.

FIRM LIMITS — not negotiable:
- No medical advice or health diagnoses. Redirect to a doctor.
- No legal or financial advice. No specific investment, trade, or money guidance.
- No hard predictions of the future — no death, no disaster, no "you will" certainties.
- No definitive relationship verdicts. Compatibility describes dynamics, never decrees.
- If a question is distressing or implies crisis, respond with care and gently \
suggest talking to a trusted person or professional.
You assist the person's own judgment — never replace it.

STYLE
Concise and substantial — usually 2-4 short paragraphs. No filler. You may use the \
app's own concepts (Life Path, Day Master, Ten Gods, the trines, the aspects). When \
relevant, point the person toward the app's Teachings for depth."""


class AskRequest(BaseModel):
    question: str
    chart_context: Optional[str] = None
    history: Optional[list] = None


@app.post("/ask")
def ask_assistant(req: AskRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="The assistant is not configured. ANTHROPIC_API_KEY is missing.",
        )
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="No question was provided.")

    # Build the message list.
    messages = []
    if req.history:
        for turn in req.history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_content = req.question.strip()
    if req.chart_context:
        user_content = (
            f"[The person's chart data, computed by AstroTeric:]\n"
            f"{req.chart_context}\n\n"
            f"[Their question:]\n{req.question.strip()}"
        )
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 900,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    headers = {
        "x-api-key": api_key.strip(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the AI service: {e}")

    if resp.status_code != 200:
        # Pass through the API's own error text so problems are visible.
        detail = resp.text[:300]
        raise HTTPException(
            status_code=502,
            detail=f"AI service returned {resp.status_code}: {detail}",
        )

    try:
        data = resp.json()
        answer = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return {"ok": True, "answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read AI response: {e}")

