import time
import logging
import random
from typing import List, Dict, Any

import openai

DEFAULT_TEMP = 0.2
DEFAULT_MAX_TOKENS = 512


def select_model() -> str:
    base = getattr(openai, "api_base", "")
    return "openrouter/auto" if base.startswith("https://openrouter.ai") else "gpt-3.5-turbo"


def chat_completion(messages: List[Dict[str, Any]], *,
                    temperature: float = DEFAULT_TEMP,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    request_timeout: int = 30,
                    retries: int = 3,
                    backoff_base: float = 0.5) -> str:
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
            stream = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                stream=True,
                request_timeout=int(request_timeout),
            )
            text = ""
            for chunk in stream:
                try:
                    delta = chunk["choices"][0]["delta"].get("content", "") if "choices" in chunk else ""
                except Exception:
                    delta = ""
                if delta:
                    text += delta
            if not text.strip():
                # non-streaming fallback same attempt
                resp = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    request_timeout=int(request_timeout),
                )
                text = resp.choices[0].message.get("content", "")
            latency = time.time() - t0
            logging.info({
                "event": "chat.complete",
                "model": model_name,
                "latency_s": round(latency, 3),
                "attempt": attempt,
                "tokens": "n/a"
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
        "retries": retries
    })
    return ""
