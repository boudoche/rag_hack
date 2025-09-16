from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

INDEX_DIR = os.getenv("INDEX_DIR", ".rag_index")
STORE_FILE = os.getenv("STORE_FILE", ".rag_index/store.jsonl")

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
CTX_CHARS_LIMIT = int(os.getenv("CTX_CHARS_LIMIT", 6000))
CHUNK_SENTENCE_AWARE = os.getenv("CHUNK_SENTENCE_AWARE", "false").lower() == "true"

SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Follow these rules strictly:\n"
    "1) Use ONLY the provided context to answer. Do not invent facts.\n"
    "2) If the answer is not fully supported by the context, say you don't know.\n"
    "3) Cite sources inline after each relevant claim as: (source: <file> [page/slide if present]).\n"
    "4) Prefer concise answers: short paragraphs and bullet points when listing.\n"
    "5) Preserve units, numbers, and dates exactly as in the sources.\n"
    "6) If multiple sources conflict, note the conflict and present both with citations.\n"
    "7) If the question is ambiguous, list the missing details you need.\n"
)
