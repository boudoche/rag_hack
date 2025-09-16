from __future__ import annotations
from typing import List, Dict, Any
from openai import OpenAI
from .config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    SYSTEM_PROMPT,
    CTX_CHARS_LIMIT,
)


def make_context(hits: List[Dict[str, Any]], limit_chars: int = CTX_CHARS_LIMIT) -> str:
    ctx = []
    used = 0
    for h in hits:
        md = h.get("metadata", {})
        src = md.get("source", "?")
        loc = (
            f" p.{md.get('page')}" if md.get("page") is not None else (
                f" slide {md.get('slide')}" if md.get("slide") is not None else ""
            )
        )
        block = f"[SOURCE: {src}{loc}]\n{h['text']}\n"
        if used + len(block) > limit_chars:
            break
        used += len(block)
        ctx.append(block)
    return "\n\n".join(ctx)


def answer_with_openai(query: str, hits: List[Dict[str, Any]], model: str = OPENAI_CHAT_MODEL) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant. Renseigne-le dans .env")
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = make_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}\n\nRéponds en citant les sources."},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
