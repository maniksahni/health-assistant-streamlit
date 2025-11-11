import time
import logging
import random
from typing import List, Dict, Any

import openai
from openai import OpenAI

DEFAULT_TEMP = 0.2
DEFAULT_MAX_TOKENS = 512


def select_model() -> str:
    base = getattr(openai, "api_base", "")
    return "openrouter/auto" if base.startswith("https://openrouter.ai") else "gpt-3.5-turbo"


def _get_client() -> OpenAI | None:
    base = getattr(openai, "api_base", "")
    api_key = getattr(openai, "api_key", None)
    try:
        if base:
            return OpenAI(api_key=api_key, base_url=base)
        return OpenAI(api_key=api_key)
    except TypeError as e:
        # Some environments may have version/env combos where the OpenAI client init receives
        # unexpected kwargs via internal wiring (e.g., proxies). Fallback to legacy path.
        logging.warning({"event": "chat.client_init_legacy_fallback", "error": str(e)})
        return None


def chat_completion(messages: List[Dict[str, Any]], *,
                    temperature: float = DEFAULT_TEMP,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    request_timeout: int = 30,
                    retries: int = 3,
                    backoff_base: float = 0.5,
                    request_id: str | None = None) -> str:
    """Call ChatCompletion with retries and jittered exponential backoff.

    Returns assistant text content or empty string on failure.
    """
    model_name = select_model()
    attempt = 0
    last_err = None
    t0 = time.time()
    while attempt <= retries:
        try:
            # Prefer streaming if possible for perceived latency
            client = _get_client()
            text = ""
            if client is not None:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    stream=True,
                    timeout=int(request_timeout),
                )
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        text += delta
                if not text.strip():
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                        timeout=int(request_timeout),
                    )
                    text = resp.choices[0].message.content or ""
            else:
                # Fallback to OpenAI 1.x module-level API (no explicit client)
                stream = openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    stream=True,
                    timeout=int(request_timeout),
                )
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        text += delta
                if not text.strip():
                    resp = openai.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                        timeout=int(request_timeout),
                    )
                    text = resp.choices[0].message.content or ""
            latency = time.time() - t0
            logging.info({
                "event": "chat.complete",
                "model": model_name,
                "latency_s": round(latency, 3),
                "attempt": attempt,
                "tokens": "n/a",
                "request_id": request_id
            })
            return text
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > retries:
                break
            # backoff with jitter
            delay = backoff_base * (2 ** (attempt - 1))
            delay = delay + random.uniform(0, delay / 2)
            time.sleep(delay)
    logging.error({
        "event": "chat.error",
        "error": str(last_err) if last_err else "unknown",
        "retries": retries,
        "request_id": request_id
    })
    return ""
