import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, cast

import requests
from duckduckgo_search import DDGS

DEFAULT_TEMP = 0.2
DEFAULT_MAX_TOKENS = 2048


def _needs_web_search(query: str) -> bool:
    """Determine if a query likely requires real-time web information."""
    query = query.lower()
    # Keywords that suggest a need for current information
    search_keywords = [
        "price of",
        "cost of",
        "how much is",
        "latest news",
        "what's new",
        "update on",
        "release date",
        "when is",
        "who is",
        "what is",
        "stock price",
        "weather",
        "forecast",
        "temperature",
        "current events",
        "in india",
        "rate of",
        "gold rate",
        "gold price",
        "weather in",
    ]
    # Check for questions about future or recent products
    if (
        "iphone 17" in query
        or "iphone 16" in query
        or "iphone 17 pro" in query
        or "17 pro" in query
    ):
        return True
    # Treat iPhone price/cost queries as real-time
    if "iphone" in query and any(
        k in query for k in ["price", "cost", "pro", "pro max", "release", "launch", "in india"]
    ):
        return True
    return any(keyword in query for keyword in search_keywords)


def _perform_web_search(query: str) -> str:
    """Perform a web search and return a formatted string of results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web search results found."

        # Format results into a string for the LLM
        formatted_results = "Web Search Results:\n"
        for i, result in enumerate(results):
            formatted_results += (
                f"[{i+1}] {result['title']}\n{result['body']}\nURL: {result['href']}\n\n"
            )
        return formatted_results
    except Exception as e:
        logging.warning(f"Web search failed: {e}")
        return "Web search failed."


def _extract_city(query: str) -> Optional[str]:
    """Extract a city name from queries like 'weather in delhi' or 'delhi weather'."""
    q = query.lower()
    m = re.search(r"weather in\s+([a-zA-Z\s]+)", q)
    if m:
        return m.group(1).strip().strip("?.,!")
    m = re.search(r"in\s+([a-zA-Z\s]+)\s+weather", q)
    if m:
        return m.group(1).strip().strip("?.,!")
    # Fallback: last word if starts with weather
    if q.startswith("weather "):
        return q.replace("weather", "", 1).strip().strip("?.,!")
    return None


def _get_weather_context(query: str) -> Optional[str]:
    """Fetch current weather using Open-Meteo for the extracted city."""
    city = _extract_city(query)
    if not city:
        return None
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params=cast(
                Dict[str, Any], {"name": city, "count": 1, "language": "en", "format": "json"}
            ),
            timeout=10,
        ).json()
        if not geo or not geo.get("results"):
            return None
        res = geo["results"][0]
        lat, lon = res.get("latitude"), res.get("longitude")
        loc_name = res.get("name")
        cc = res.get("country")
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=cast(
                Dict[str, Any],
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
            ),
            timeout=10,
        ).json()
        # Handle both current and current_weather styles
        cur = wx.get("current") or wx.get("current_weather") or {}
        t = cur.get("temperature_2m") or cur.get("temperature")
        rh = cur.get("relative_humidity_2m")
        wcode = cur.get("weather_code")
        wind = cur.get("wind_speed_10m") or cur.get("windspeed")
        parts = [f"Location: {loc_name}, {cc}"]
        if t is not None:
            parts.append(f"Temperature: {t}°C")
        if rh is not None:
            parts.append(f"Humidity: {rh}%")
        if wind is not None:
            parts.append(f"Wind: {wind} km/h")
        if wcode is not None:
            parts.append(f"Weather code: {wcode}")
        return "Current Weather (Open-Meteo):\n" + " | ".join(parts)
    except Exception as e:
        logging.warning(f"Weather fetch failed: {e}")
        return None


def _get_gold_context(query: str) -> Optional[str]:
    """Fetch spot gold price (USD) using metals.live; best-effort, fallback to None."""
    if not any(
        k in query.lower() for k in ["gold rate", "rate of gold", "gold price", "price of gold"]
    ):
        return None
    try:
        # Primary endpoint
        r = requests.get("https://api.metals.live/v1/spot/gold", timeout=10)
        data = r.json()
        # Expected: list of dicts or list of lists with price
        price = None
        if isinstance(data, list):
            # e.g., [{"gold": 2371.23, "timestamp": ...}, ...] or [["gold", 2371.23], ...]
            last = data[-1]
            if isinstance(last, dict):
                price = last.get("gold") or last.get("price")
            elif isinstance(last, list) and len(last) >= 2:
                price = last[1]
        if price:
            return f"Spot Gold (XAU) price ~ ${price} per troy ounce (source: metals.live)"
    except Exception as e:
        logging.warning(f"Gold price fetch failed: {e}")
    return None


def _model_candidates(provider: str) -> list[str]:
    # Return valid model IDs for the target provider
    provider = (provider or "").lower()
    if provider == "openrouter":
        return [
            "mistralai/mistral-nemo:free",
            "mistralai/mistral-nemo",
        ]
    # Default to OpenAI-compatible IDs
    return [
        "gpt-4o-mini",
    ]


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 60,
    retries: int = 3,
    backoff_base: float = 0.5,
    request_id: Optional[str] = None,
    api_key: Optional[str] = None,
    preferred_models: Optional[List[str]] = None,
) -> str:
    """Call the appropriate chat API directly with retries.

    Returns assistant text content or an error message string on failure.
    """
    import os

    import streamlit as st

    used_or = False
    try:
        if not api_key:
            k = st.secrets.get("OPENROUTER_API_KEY")
            if k:
                api_key = k
                used_or = True
        if not api_key:
            k = st.secrets.get("OPENAI_API_KEY")
            if k:
                api_key = k
    except Exception:
        pass
    if not api_key:
        k = os.getenv("OPENROUTER_API_KEY")
        if k:
            api_key = k
            used_or = True
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: No API key configured"

    # Determine provider from environment or API key prefix
    env_provider = (os.getenv("OPENAI_PROVIDER") or "").strip().lower()
    provider = "openai"
    if env_provider == "openrouter" or used_or or str(api_key).startswith("sk-or-"):
        provider = "openrouter"

    # Determine base URL
    base_url = "https://api.openai.com/v1"
    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"

    candidates = _model_candidates(provider)
    attempt = 0
    last_err = None
    t0 = time.time()

    # Add specialized factual context (weather/gold) or fall back to web search
    if messages and messages[-1]["role"] == "user":
        last_user_message = messages[-1]["content"]
        special_ctx = _get_weather_context(last_user_message) or _get_gold_context(
            last_user_message
        )
        if special_ctx:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Use the following factual data to answer concisely:\n\n{special_ctx}",
                },
            )
        elif _needs_web_search(last_user_message):
            logging.info(f"Performing web search for query: {last_user_message}")
            search_results = _perform_web_search(last_user_message)
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Please use the following web search results to answer the user's question. If the results are not relevant, inform the user that you couldn't find current information.\n\n{search_results}",
                },
            )

    # Context already inserted above if found

    # Allow explicit override via environment
    env_model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL")
    models = []
    if env_model and isinstance(env_model, str):
        m = env_model.strip()
        bad = False
        s = m.lower()
        # Treat URLs or api path fragments as invalid model IDs
        if s.startswith("http://") or s.startswith("https://"):
            bad = True
        if "api/v1" in s:
            bad = True
        if not bad and m:
            models = [m]
    if not models:
        models = candidates
    # Allow explicit override from caller (e.g., UI selection)
    if preferred_models and isinstance(preferred_models, list):
        try:
            _pm = [str(m).strip() for m in preferred_models if str(m).strip()]
            if _pm:
                models = _pm
        except Exception:
            pass

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://health-assistant-app-58hi.onrender.com",
        "X-Title": "Health Assistant",
    }

    while attempt <= retries:
        try:
            text = ""
            last_non_200 = ""

            for model_name in models:
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "stream": False,  # Using non-streaming for reliability
                }

                try:
                    response = requests.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=request_timeout,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        text = result["choices"][0]["message"]["content"] or ""
                        if text.strip():
                            break
                    else:
                        try:
                            body = response.text
                        except Exception:
                            body = ""
                        last_non_200 = (
                            f"Error: HTTP {response.status_code} for {model_name}: {body[:200]}"
                        )
                        logging.warning(f"Model {model_name} failed: {response.status_code}")
                        continue

                except Exception as e_model:
                    logging.warning(f"Model {model_name} error: {e_model}")
                    continue

            latency = time.time() - t0
            logging.info(
                {
                    "event": "chat.complete",
                    "model": model_name,
                    "latency_s": round(latency, 3),
                    "attempt": attempt,
                    "tokens": "n/a",
                    "request_id": request_id,
                }
            )

            t = text.strip()
            if not t and last_non_200:
                return last_non_200
            return t

        except Exception as e:
            last_err = e
            attempt += 1
            if attempt <= retries:
                # Exponential backoff with jitter
                delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                break

    logging.error({"event": "chat.failed", "error": str(last_err)})
    return "I'm experiencing technical difficulties. Please try again in a moment."


def chat_completion_stream(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 60,
    retries: int = 0,
    backoff_base: float = 0.5,
    request_id: Optional[str] = None,
    api_key: Optional[str] = None,
    preferred_models: Optional[List[str]] = None,
):
    """Yield assistant text chunks using OpenAI/OpenRouter streaming. Falls back to non-streaming."""

    # Resolve API key similar to non-streaming path
    used_or = False
    try:
        if not api_key:
            k = None
            try:
                import streamlit as st  # type: ignore

                k = st.secrets.get("OPENROUTER_API_KEY")
            except Exception:
                k = None
            if k:
                api_key = k
                used_or = True
        if not api_key:
            try:
                import streamlit as st  # type: ignore

                k = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                k = None
            if k:
                api_key = k
    except Exception:
        pass
    if not api_key:
        k = os.getenv("OPENROUTER_API_KEY")
        if k:
            api_key = k
            used_or = True
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key: surface a single error chunk and stop
        yield "Error: No API key configured"
        return

    env_provider = (os.getenv("OPENAI_PROVIDER") or "").strip().lower()
    provider = "openai"
    if env_provider == "openrouter" or used_or or str(api_key).startswith("sk-or-"):
        provider = "openrouter"

    base_url = "https://api.openai.com/v1"
    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"

    candidates = _model_candidates(provider)
    model = None
    if preferred_models and isinstance(preferred_models, list) and preferred_models:
        model = str(preferred_models[0])
    elif os.getenv("OPENAI_MODEL") or os.getenv("MODEL"):
        _env_model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL")
        model = _env_model.strip() if isinstance(_env_model, str) else candidates[0]
    else:
        model = candidates[0] if candidates else "gpt-4o-mini"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://health-assistant-app-58hi.onrender.com",
        "X-Title": "Health Assistant",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }

    try:
        with requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=request_timeout,
            stream=True,
        ) as r:
            if r.status_code != 200:
                # Fall back to non-streaming
                yield chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                    retries=retries,
                    backoff_base=backoff_base,
                    request_id=request_id,
                    api_key=api_key,
                    preferred_models=[model],
                )
                return
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                try:
                    delta = evt.get("choices", [{}])[0].get("delta", {}).get("content")
                    if not delta:
                        # Some providers embed full message content
                        delta = evt.get("choices", [{}])[0].get("message", {}).get("content")
                    if delta:
                        yield str(delta)
                except Exception:
                    continue
    except Exception:
        # Any failure: fall back to one-shot
        yield chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            retries=retries,
            backoff_base=backoff_base,
            request_id=request_id,
            api_key=api_key,
            preferred_models=[model],
        )
