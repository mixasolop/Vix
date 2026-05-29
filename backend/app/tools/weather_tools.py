import asyncio
from datetime import date, timedelta
import json
import logging
from urllib.parse import urlencode
from urllib.request import urlopen

from app.schemas.tools import ToolResult

LOGGER = logging.getLogger("app.tools.weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_CONDITIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def get_weather(arguments: dict[str, object]) -> ToolResult:
    location = str(arguments.get("location", "")).strip()
    requested_date = str(arguments.get("date", "today")).strip().lower()
    if not location:
        return ToolResult(tool="get_weather", status="failed", error="Location is required.")

    try:
        return await asyncio.to_thread(_get_weather_sync, location, requested_date)
    except Exception as exc:
        LOGGER.exception("get_weather failed | location=%s | date=%s", location, requested_date)
        return ToolResult(tool="get_weather", status="failed", error=f"Weather lookup failed: {exc}")


def _get_weather_sync(location: str, requested_date: str) -> ToolResult:
    target_date = _resolve_requested_date(requested_date)
    place = _geocode(location)
    forecast = _forecast(place["latitude"], place["longitude"], target_date)
    daily = forecast["daily"]
    index = daily["time"].index(target_date.isoformat())

    condition = WEATHER_CODE_CONDITIONS.get(int(daily["weather_code"][index]), "Unknown")
    temperature = {
        "min_c": daily["temperature_2m_min"][index],
        "max_c": daily["temperature_2m_max"][index],
    }
    if target_date == date.today() and forecast.get("current") is not None:
        temperature["current_c"] = forecast["current"].get("temperature_2m")

    precipitation_probability = daily.get("precipitation_probability_max", [None])[index]
    wind = {
        "max_kmh": daily["wind_speed_10m_max"][index],
    }
    if target_date == date.today() and forecast.get("current") is not None:
        wind["current_kmh"] = forecast["current"].get("wind_speed_10m")

    resolved_location = ", ".join(
        part
        for part in (place.get("name"), place.get("admin1"), place.get("country"))
        if part
    )
    message = (
        f"Weather for {resolved_location} on {target_date.isoformat()}: {condition}, "
        f"{temperature['min_c']} to {temperature['max_c']} C, "
        f"precipitation probability {precipitation_probability}%, wind up to {wind['max_kmh']} km/h."
    )

    return ToolResult(
        tool="get_weather",
        status="success",
        output={
            "status": "success",
            "message": message,
            "location": resolved_location,
            "date": target_date.isoformat(),
            "temperature": temperature,
            "condition": condition,
            "precipitation_probability": precipitation_probability,
            "wind": wind,
            "source": "Open-Meteo",
        },
    )


def _resolve_requested_date(requested_date: str) -> date:
    if requested_date in {"", "today"}:
        return date.today()
    if requested_date == "tomorrow":
        return date.today() + timedelta(days=1)
    return date.fromisoformat(requested_date)


def _geocode(location: str) -> dict[str, object]:
    payload = _get_json(
        GEOCODING_URL,
        {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Location was not found: {location}")
    return results[0]


def _forecast(latitude: object, longitude: object, target_date: date) -> dict[str, object]:
    days_ahead = max((target_date - date.today()).days + 1, 1)
    payload = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
            "forecast_days": min(max(days_ahead, 1), 16),
            "timezone": "auto",
        },
    )
    daily = payload.get("daily") or {}
    if target_date.isoformat() not in daily.get("time", []):
        raise ValueError(f"Forecast date is not available: {target_date.isoformat()}")
    return payload


def _get_json(url: str, params: dict[str, object]) -> dict[str, object]:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))
