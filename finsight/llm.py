"""LLM fallback chain.

Order: PRIMARY_MODEL -> FALLBACK_MODELS -> Groq (if GROQ_API_KEY) -> Gemini (if GEMINI_API_KEY).
Local Ollama models are only included if they are actually installed, and the
whole Ollama tier is skipped if the server is down.
"""
import logging
import os

import requests
from crewai import LLM

from . import config

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def ollama_models() -> set:
    try:
        tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json()["models"]
        names = {m["name"] for m in tags}
        return names | {n.removesuffix(":latest") for n in names}
    except Exception:  # noqa: BLE001
        return set()


def model_chain() -> list:
    installed = ollama_models()
    chain = []
    for m in [config.PRIMARY_MODEL, *config.FALLBACK_MODELS]:
        if m.startswith("ollama/") and m.split("/", 1)[1] not in installed:
            continue
        if m not in chain:
            chain.append(m)
    if os.getenv("GROQ_API_KEY") and config.GROQ_MODEL not in chain:
        chain.append(config.GROQ_MODEL)
    if os.getenv("GEMINI_API_KEY") and config.GEMINI_MODEL not in chain:
        chain.append(config.GEMINI_MODEL)
    return chain


def make_llm(model: str) -> LLM:
    if model.startswith("ollama/"):
        kwargs = {"base_url": config.OLLAMA_BASE_URL, "temperature": 0.1, "max_tokens": 4096, "timeout": 600}
        # Qwen3 "thinking" makes every call ~7x slower; switch it off unless asked for.
        if "qwen3" in model and os.getenv("QWEN_THINKING", "false").lower() != "true":
            kwargs["reasoning_effort"] = "none"
        return LLM(model=model, **kwargs)
    kwargs = {"temperature": 0.1, "max_tokens": 4096}
    # gpt-oss also "thinks" before answering; keep it short so cloud runs stay fast.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = os.getenv("GPT_OSS_REASONING", "low")
    if model.startswith("groq/") and "qwen" in model:
        # Groq's free plan caps Qwen at 1,000 output tokens a minute; a larger request is refused outright.
        kwargs["max_tokens"] = int(os.getenv("GROQ_QWEN_MAX_TOKENS", "900"))
    if model.startswith("groq/"):
        # Groq speaks the OpenAI API, so use CrewAI's native OpenAI client against Groq's
        # endpoint (the LiteLLM route sends a cache field that Groq rejects).
        return LLM(model=model.removeprefix("groq/"), provider="openai", base_url=GROQ_BASE_URL,
                   api_key=os.getenv("GROQ_API_KEY"), **kwargs)
    return LLM(model=model, **kwargs)
