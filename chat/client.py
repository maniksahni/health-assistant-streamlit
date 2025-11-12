import logging
import random
import time
from typing import Any, Dict, List, Optional

import requests
from duckduckgo_search import DDGS

DEFAULT_TEMP = 0.2
DEFAULT_MAX_TOKENS = 512


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
        "current events",
        "in india",
        "rate of",
        "gold rate",
        "gold price",
        "weather in",
    ]
    # Check for questions about future or recent products
    if "iphone 17" in query or "iphone 16" in query or "iphone 17 pro" in query or "17 pro" in query:
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


def _model_candidates(provider: str) -> list[str]:
    # Return valid model IDs for the target provider
    provider = (provider or "").lower()
    if provider == "openrouter":
        return [
            "mistralai/mistral-nemo:free",
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

    # Check if the last user message requires a web search
    search_context_message = None
    if messages and messages[-1]["role"] == "user":
        last_user_message = messages[-1]["content"]
        if _needs_web_search(last_user_message):
            logging.info(f"Performing web search for query: {last_user_message}")
            search_results = _perform_web_search(last_user_message)
            # Create a system message with the search results as context
            search_context_message = {
                "role": "system",
                "content": f"Please use the following web search results to answer the user's question. If the results are not relevant, inform the user that you couldn't find current information.\n\n{search_results}",
            }

    # If search was performed, add the context to the start of the message list
    if search_context_message:
        messages.insert(0, search_context_message)

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
