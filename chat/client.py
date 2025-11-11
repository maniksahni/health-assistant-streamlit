import time
import logging
import random
import requests
from typing import List, Dict, Any, Optional

DEFAULT_TEMP = 0.2
DEFAULT_MAX_TOKENS = 512


def _model_candidates(provider: str) -> list[str]:
    # Return valid model IDs for the target provider
    provider = (provider or '').lower()
    if provider == 'openrouter':
        return [
            "mistralai/mistral-nemo",
        ]
    # Default to OpenAI-compatible IDs
    return [
        "gpt-4o-mini",
    ]


def chat_completion(messages: List[Dict[str, Any]], *,
                    temperature: float = DEFAULT_TEMP,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    request_timeout: int = 60,
                    retries: int = 3,
                    backoff_base: float = 0.5,
                    request_id: Optional[str] = None,
                    api_key: Optional[str] = None) -> str:
    """Call the appropriate chat API directly with retries.

    Returns assistant text content or an error message string on failure.
    """
    import os

    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "Error: No API key configured"

    # Determine provider from environment or API key prefix
    env_provider = (os.getenv('OPENAI_PROVIDER') or '').strip().lower()
    provider = 'openai'
    if env_provider == 'openrouter':
        provider = 'openrouter'
    elif str(api_key).startswith("sk-or-"):
        provider = 'openrouter'

    # Determine base URL
    base_url = "https://api.openai.com/v1"
    if provider == 'openrouter':
        base_url = "https://openrouter.ai/api/v1"

    candidates = _model_candidates(provider)
    attempt = 0
    last_err = None
    t0 = time.time()

    # Allow explicit override via environment
    env_model = os.getenv('OPENAI_MODEL') or os.getenv('MODEL')
    models = []
    if env_model and isinstance(env_model, str):
        m = env_model.strip()
        bad = False
        s = m.lower()
        # Treat URLs or api path fragments as invalid model IDs
        if s.startswith('http://') or s.startswith('https://'):
            bad = True
        if 'api/v1' in s:
            bad = True
        if not bad and m:
            models = [m]
    if not models:
        models = candidates
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://health-assistant.streamlit.app',
        'X-Title': 'Health Assistant'
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
                    "stream": False  # Using non-streaming for reliability
                }
                
                try:
                    response = requests.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=request_timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result['choices'][0]['message']['content'] or ""
                        if text.strip():
                            break
                    else:
                        try:
                            body = response.text
                        except Exception:
                            body = ""
                        last_non_200 = f"Error: HTTP {response.status_code} for {model_name}: {body[:200]}"
                        logging.warning(f"Model {model_name} failed: {response.status_code}")
                        continue
                        
                except Exception as e_model:
                    logging.warning(f"Model {model_name} error: {e_model}")
                    continue
            
            latency = time.time() - t0
            logging.info({
                "event": "chat.complete",
                "model": model_name,
                "latency_s": round(latency, 3),
                "attempt": attempt,
                "tokens": "n/a",
                "request_id": request_id
            })
            
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
