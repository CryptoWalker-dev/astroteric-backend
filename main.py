"""
AstroTeric — Backend (Ephemeris + Assistant)
=============================================
A small FastAPI server with two jobs:

  1. EPHEMERIS  — compute true planetary positions, the Ascendant
                  (rising sign), and the Midheaven (Swiss Ephemeris).
  2. ASSISTANT  — answer questions about a chart, or general esoteric
                  questions, using Claude. A grounded practitioner's
                  tool: honest, never sugar-coated, knows its limits.

Endpoints:
  GET  /        — health check
  POST /chart   — compute a natal chart from birth data
  POST /ask     — ask the AstroTeric assistant a question
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import swisseph as swe
import anthropic

app = FastAPI(title="AstroTeric Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Signs and planets ──────────────────────────────────────────────
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
        "version": "2.0.0",
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
# ASSISTANT — the AstroTeric bot
# ════════════════════════════════════════════════════════════════

# The assistant's character. This is the heart of the bot — it defines
# how it behaves. Grounded, honest, never sugar-coated; a tool for the
# practitioner and the serious newcomer alike.
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

FIRM LIMITS — these are not negotiable:
- No medical advice or health diagnoses. Redirect to a doctor.
- No legal or financial advice. No specific investment, trade, or money guidance.
- No hard predictions of the future — no death, no disaster, no "you will" certainties.
- No definitive relationship verdicts ("you must leave," "you are doomed together"). \
Compatibility describes dynamics, never decrees outcomes.
- If a question is distressing or implies crisis, respond with care and gently \
suggest talking to a trusted person or professional.
You are an assistant to the person's own judgment — never a replacement for it. The \
reader, or the person, decides; you inform.

STYLE
Concise and substantial — usually 2-4 short paragraphs. No filler. You may use the \
app's own concepts (Life Path, Day Master, Ten Gods, the trines, the aspects). When \
relevant, point the person toward the app's Teachings for depth."""


class AskRequest(BaseModel):
    question: str
    # Optional chart context — the app sends what it has computed.
    chart_context: Optional[str] = None
    # Optional short prior exchange, for follow-up questions.
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

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build the conversation. Prior history first (if any), then the
        # current question with its chart context attached.
        messages = []
        if req.history:
            for turn in req.history[-6:]:  # cap history length
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

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # Collect the text from the response.
        answer = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return {"ok": True, "answer": answer.strip()}

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Assistant error: {e.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {e}")
