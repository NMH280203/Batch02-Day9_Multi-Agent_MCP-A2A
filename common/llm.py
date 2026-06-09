"""Shared LLM factory for all agents.

Uses the OpenAI API via LangChain's ChatOpenAI client.
Model is configurable via the OPENAI_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenAI."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
    )
