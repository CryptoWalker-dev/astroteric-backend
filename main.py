
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
  POST /chart   — compute a natal chart from birth data (tropical / Western)
  POST /jyotish — compute a sidereal Vedic chart: nakshatra + Vimshottari dasha
  POST /ask     — ask the AstroTeric assistant a question
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
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
# JYOTISH (Vedic) — sidereal chart, Moon's nakshatra, Vimshottari dasha
# ════════════════════════════════════════════════════════════════
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha",
    "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
    "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]
DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17,
}
VEDIC_PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY,
    "Venus": swe.VENUS, "Mars": swe.MARS, "Jupiter": swe.JUPITER, "Saturn": swe.SATURN,
}


@app.post("/jyotish")
def compute_jyotish(req: ChartRequest):
    try:
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

        local = datetime(
            req.year, req.month, req.day, req.hour, req.minute,
            tzinfo=timezone(timedelta(hours=req.tz_offset)),
        )
        ut = local.astimezone(timezone.utc)
        jd = swe.julday(
            ut.year, ut.month, ut.day,
            ut.hour + ut.minute / 60.0 + ut.second / 3600.0,
        )

        flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
        planets = {}
        moon_lon = None
        for name, code in VEDIC_PLANETS.items():
            result, _ = swe.calc_ut(jd, code, flags)
            lon = result[0] % 360.0
            planets[name] = sign_of(lon)
            if name == "Moon":
                moon_lon = lon

        try:
            _cusps, ascmc = swe.houses_ex(
                jd, req.latitude, req.longitude, b"P", swe.FLG_SIDEREAL
            )
            asc = ascmc[0] % 360.0
        except Exception:
            _houses, ascmc = swe.houses(jd, req.latitude, req.longitude, b"P")
            asc = (ascmc[0] - swe.get_ayanamsa_ut(jd)) % 360.0
        lagna = sign_of(asc)

        span = 360.0 / 27.0
        nak_index = int(moon_lon // span) % 27
        pos_in_nak = moon_lon - nak_index * span
        pada = int(pos_in_nak // (span / 4.0)) + 1
        nak_lord = DASHA_LORDS[nak_index % 9]
        moon_nakshatra = {
            "nakshatra": NAKSHATRAS[nak_index],
            "pada": pada,
            "lord": nak_lord,
        }

        frac_traversed = pos_in_nak / span
        start_i = DASHA_LORDS.index(nak_lord)
        balance = (1.0 - frac_traversed) * DASHA_YEARS[nak_lord]
        seq = []
        age = 0.0
        for k in range(9):
            lord = DASHA_LORDS[(start_i + k) % 9]
            full = DASHA_YEARS[lord]
            dur = balance if k == 0 else full
            seq.append({
                "lord": lord,
                "start_age": round(age, 2),
                "end_age": round(age + dur, 2),
                "full_period": full,
                "partial": (k == 0),
            })
            age += dur

        return {
            "ok": True,
            "julian_day": round(jd, 6),
            "ayanamsa": round(swe.get_ayanamsa_ut(jd), 4),
            "lagna": lagna,
            "planets": planets,
            "moon_nakshatra": moon_nakshatra,
            "dasha": {"sequence": seq},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Jyotish calculation failed: {e}")


# ════════════════════════════════════════════════════════════════
# ASSISTANT — the AstroTeric bot
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the Oracle — the in-app guide of AstroTeric, an esoteric \
reference app that computes and teaches numerology (classical Pythagorean and \
Chaldean), Western astrology, Vedic astrology (Jyotish), Chinese astrology and BaZi \
(Four Pillars), tarot, palmistry, and related traditions. You answer questions from \
people using the app — about their own chart, or about any concept in these systems.

GROUND TRUTH ABOUT THE APP
- You may be given a CHART CONTEXT block describing what AstroTeric contains and the \
current user's computed placements. Treat it as authoritative about the app.
- NEVER tell a user that a system, feature, or topic is "not in the app" if it appears \
in that context or in the list above. AstroTeric FULLY includes Vedic astrology / \
Jyotish — nakshatras, padas, the Lagna, and the Vimshottari dasha — alongside \
numerology, Western astrology, Chinese astrology and BaZi, tarot, palmistry, and a \
large Teachings library.
- If a user's specific data for some system is not shown in the provided context, it \
simply has not been computed in this session yet — it computes in the Reading tab once \
a birth time and place are set. Say that and point them there. Do NOT deny the feature \
exists.
- When a topic runs deep, point the user to its entry in the app's Teachings, and to \
Your Daily (day energies) or the Date Explorer (choosing good-energy days) where useful.

YOUR ROLE
A practitioner's tool and a guide for serious newcomers. Two kinds of questions: (A) \
questions about a specific chart, when chart data is provided, and (B) general \
questions about esoteric systems and their concepts. Answer both.

YOUR CHARACTER
- Honest above all. Never sugar-coat, never give false hope, never inflate. Name the \
shadow as readily as the gift. People come to you for the real thing, not flattery.
- Grounded. When chart data is provided, base your answer on THAT data, using the \
person's actual numbers and placements by name. Do not invent placements.
- Clear. Explain esoteric terms in plain language. A newcomer should understand you; a \
practitioner should respect you.
- Humble about the nature of this knowledge. These are interpretive traditions, not \
deterministic science. Frame readings as lenses for reflection, to be tested against \
real life — never as fixed fate or prediction.

A NOTE ON NUMEROLOGY METHOD
The Life Path can be computed two classical ways (digit-sum and component); they agree \
for most people and diverge only around master numbers. If the context notes a \
divergence, explain both honestly and note they share the same root — the lived \
reading holds either way. The app lets the user set their preferred method in Settings.

FIRM LIMITS — these are not negotiable:
- No medical advice or health diagnoses. Redirect to a doctor.
- No legal or financial advice. No specific investment, trade, or money guidance.
- No hard predictions of the future — no death, no disaster, no "you will" certainties.
- No definitive relationship verdicts ("you must leave," "you are doomed together"). \
Compatibility describes dynamics, never decrees outcomes.
- If a question is distressing or implies crisis, respond with care and gently suggest \
talking to a trusted person or professional.
You are an assistant to the person's own judgment — never a replacement for it. The \
person decides; you inform.

STYLE
Concise and substantial — usually 2-4 short paragraphs, plain readable prose, no \
filler. Use the app's own concepts (Life Path, Day Master, nakshatra, dasha, the \
trines, the aspects) and the person's own placements by name. When relevant, point \
toward the app's Teachings for depth."""


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

    try:
        # Explicit HTTP client with a generous timeout and a plain transport.
        # This avoids the SDK's custom transport, the known cause of spurious
        # "Connection error" failures on some hosts.
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0))
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)

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

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        answer = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return {"ok": True, "answer": answer.strip()}

    except anthropic.APIStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Assistant error {e.status_code}: {getattr(e, 'message', str(e))}",
        )
    except Exception as e:
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        detail = f"{type(e).__name__}: {e}"
        if cause:
            detail = f"{detail}  (underlying: {type(cause).__name__}: {cause})"
        raise HTTPException(status_code=500, detail=f"Assistant failed: {detail}")

     
