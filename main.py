
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

app = FastAPI(title="AstroTeric Backend", version="3.0.0")

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
    if index > 11:
        index = 11
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
        "version": "3.0.0",
        "ephemeris": True,
        "jyotish": True,
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
# JYOTISH — Vedic (sidereal) chart
#
# Jyotish uses the SIDEREAL zodiac, aligned to the fixed stars,
# rather than the tropical zodiac of Western astrology. The two
# differ by the "ayanamsha" (~24 degrees). The standard ayanamsha
# for Jyotish is Lahiri. This endpoint also computes the Moon's
# nakshatra — one of the 27 lunar mansions, central to Jyotish.
# ════════════════════════════════════════════════════════════════

# The 27 nakshatras, in order from 0 degrees sidereal Aries.
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha",
    "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha",
    "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati",
]

# The dasha lord sequence (Vimshottari), and each lord's period in years.
DASHA_LORDS = [
    ("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10), ("Mars", 7),
    ("Rahu", 18), ("Jupiter", 16), ("Saturn", 19), ("Mercury", 17),
]
# Each nakshatra is ruled by a dasha lord, cycling through the 9 lords.
NAKSHATRA_LORD_ORDER = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury",
]


def nakshatra_of(moon_sidereal_lon: float) -> dict:
    """Given the Moon's sidereal longitude, find its nakshatra and pada."""
    lon = moon_sidereal_lon % 360.0
    span = 360.0 / 27.0          # 13 deg 20 min per nakshatra
    # A tiny epsilon guards against floating-point error at exact
    # boundaries (e.g. 120.0 / span computing as 8.9999 instead of 9).
    index = int((lon + 1e-9) // span)
    if index > 26:
        index = 26
    position_in = lon - index * span
    pada = int((position_in + 1e-9) // (span / 4.0)) + 1
    if pada > 4:
        pada = 4
    lord = NAKSHATRA_LORD_ORDER[index % 9]
    return {
        "nakshatra": NAKSHATRAS[index],
        "index": index,
        "pada": pada,
        "lord": lord,
        "position": round(position_in, 3),
        "span": span,
    }


def compute_vimshottari(moon_sidereal_lon: float, birth_jd: float):
    """Compute the Vimshottari dasha sequence from the Moon's nakshatra.
    Returns the starting dasha and the sequence of mahadasha periods."""
    nak = nakshatra_of(moon_sidereal_lon)
    span = nak["span"]
    # Fraction of the nakshatra already traversed at birth.
    fraction_done = nak["position"] / span
    # Find the starting lord and its total period.
    start_lord = nak["lord"]
    lord_names = [l[0] for l in DASHA_LORDS]
    start_idx = lord_names.index(start_lord)
    start_period = DASHA_LORDS[start_idx][1]
    # Balance of the first dasha remaining at birth.
    balance_years = start_period * (1.0 - fraction_done)

    # Build the mahadasha sequence: first the balance, then full periods.
    sequence = []
    age = 0.0
    # First (partial) dasha
    sequence.append({
        "lord": start_lord,
        "start_age": 0.0,
        "end_age": round(balance_years, 2),
        "full_period": start_period,
        "partial": True,
    })
    age = balance_years
    # The following eight full dashas
    for i in range(1, 9):
        lord, period = DASHA_LORDS[(start_idx + i) % 9]
        sequence.append({
            "lord": lord,
            "start_age": round(age, 2),
            "end_age": round(age + period, 2),
            "full_period": period,
            "partial": False,
        })
        age += period
    return {"starting_lord": start_lord, "sequence": sequence}


@app.post("/jyotish")
def compute_jyotish(req: ChartRequest):
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

        # Switch Swiss Ephemeris into SIDEREAL mode, Lahiri ayanamsha.
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)
        flag = swe.FLG_SWIEPH | swe.FLG_SIDEREAL

        # Sidereal planetary positions.
        planets = {}
        moon_lon = None
        for name, code in PLANETS.items():
            result, _ = swe.calc_ut(jd, code, flag)
            lon = result[0]
            if name == "Moon":
                moon_lon = lon
            info = sign_of(lon)
            info["retrograde"] = result[3] < 0
            planets[name] = info

        # Sidereal houses / Lagna (the Vedic rising sign).
        houses, ascmc = swe.houses_ex(
            jd, req.latitude, req.longitude, b"W", flag
        )
        lagna = sign_of(ascmc[0])

        # The ayanamsha value used (for transparency).
        ayanamsha = swe.get_ayanamsa_ut(jd)

        # Nakshatra of the Moon, and the Vimshottari dasha sequence.
        nak = nakshatra_of(moon_lon) if moon_lon is not None else None
        dasha = compute_vimshottari(moon_lon, jd) if moon_lon is not None else None

        # Reset to tropical so the /chart endpoint is unaffected.
        swe.set_sid_mode(swe.SIDM_FAGAN_BRADLEY, 0, 0)

        return {
            "ok": True,
            "system": "Jyotish (sidereal, Lahiri ayanamsha)",
            "julian_day": round(jd, 6),
            "ayanamsha": round(ayanamsha, 4),
            "planets": planets,
            "lagna": lagna,
            "moon_nakshatra": nak,
            "dasha": dasha,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Jyotish calculation failed: {e}")


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
