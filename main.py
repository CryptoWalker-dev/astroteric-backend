"""
AstroTeric — Ephemeris Backend
================================
A small FastAPI server that computes true planetary positions, the
Ascendant (rising sign), and the Midheaven using the Swiss Ephemeris.

This is the "backend" for the AstroTeric app. The app sends a birth
date, time, and location; this server returns the real astronomical
chart. Later, an AI-assistant endpoint can be added to this same file.

Endpoints:
  GET  /              — health check, confirms the server is alive
  POST /chart         — compute a full natal chart from birth data
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import swisseph as swe

app = FastAPI(title="AstroTeric Ephemeris", version="1.0.0")

# ── CORS ───────────────────────────────────────────────────────────
# The app (served from Netlify) runs in a browser on a different
# domain than this server, so the browser will block requests unless
# the server explicitly allows them. "*" allows any origin — fine for
# now; can be tightened to just your Netlify domain later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── The twelve signs, in zodiacal order from 0° Aries ──────────────
SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Swiss Ephemeris planet constants we report
PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}


def sign_of(longitude: float) -> dict:
    """Convert an ecliptic longitude (0-360°) to a sign + degree."""
    longitude = longitude % 360.0
    index = int(longitude // 30)
    degree_in_sign = longitude - index * 30
    return {
        "sign": SIGNS[index],
        "degree": round(degree_in_sign, 2),
        "longitude": round(longitude, 4),
    }


# ── Request / response shapes ──────────────────────────────────────
class ChartRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int = 12
    minute: int = 0
    # Timezone offset from UTC in hours. Phoenix is -7 (no daylight saving).
    tz_offset: float = 0.0
    # Geographic coordinates. Positive = North / East.
    latitude: float
    longitude: float


@app.get("/")
def health():
    """Simple health check — confirms the server is running."""
    return {
        "status": "alive",
        "service": "AstroTeric Ephemeris",
        "version": "1.0.0",
    }


@app.post("/chart")
def compute_chart(req: ChartRequest):
    """Compute a full natal chart: planets, Ascendant, Midheaven."""
    try:
        # Convert the local birth time to Universal Time (UT).
        local = datetime(
            req.year, req.month, req.day,
            req.hour, req.minute,
            tzinfo=timezone(timedelta(hours=req.tz_offset)),
        )
        ut = local.astimezone(timezone.utc)

        # Swiss Ephemeris works in Julian Day. Build it from UT.
        jd = swe.julday(
            ut.year, ut.month, ut.day,
            ut.hour + ut.minute / 60.0 + ut.second / 3600.0,
        )

        # ── Planetary positions ──
        planets = {}
        for name, code in PLANETS.items():
            result, _ = swe.calc_ut(jd, code)
            longitude = result[0]
            speed = result[3]  # daily motion; negative = retrograde
            info = sign_of(longitude)
            info["retrograde"] = speed < 0
            planets[name] = info

        # ── Houses, Ascendant, Midheaven ──
        # 'P' = Placidus house system, the most common in Western astrology.
        houses, ascmc = swe.houses(
            jd, req.latitude, req.longitude, b"P"
        )
        ascendant = sign_of(ascmc[0])   # the rising sign
        midheaven = sign_of(ascmc[1])   # the MC

        house_cusps = [sign_of(h) for h in houses]

        return {
            "ok": True,
            "input": req.dict(),
            "julian_day": round(jd, 6),
            "planets": planets,
            "ascendant": ascendant,
            "midheaven": midheaven,
            "houses": house_cusps,
        }

    except Exception as e:
        # Never crash silently — return a clear error the app can show.
        raise HTTPException(status_code=400, detail=f"Chart calculation failed: {e}")
