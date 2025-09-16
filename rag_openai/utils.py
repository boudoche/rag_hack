from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List, Iterable
import re


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def read_text_file(p: Path) -> str:
    return p.read_text(errors="ignore", encoding="utf-8")


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Simple multilingual-friendly split on sentence enders and newlines
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    return [s.strip() for s in parts if s and s.strip()]


def chunk_text(
    text: str,
    max_tokens: int = 1000,
    overlap: int = 150,
    use_sentences: bool = True,
) -> List[str]:
    # Approx tokens ≈ words
    if not text or not text.strip():
        return []
    if not use_sentences:
        words = text.split()
        if not words:
            return []
        step = max(1, max_tokens - overlap)
        chunks: List[str] = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + max_tokens])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # Sentence-aware packing to target max_tokens
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_count = 0
    for sent in sentences:
        wc = len(sent.split())
        if current and current_count + wc > max_tokens:
            chunks.append(" ".join(current))
            # Start new chunk with overlap: carry tail from previous
            if overlap > 0 and chunks[-1]:
                tail_words = chunks[-1].split()
                carry = tail_words[max(0, len(tail_words) - overlap) :]
                current = [" ".join(carry)] if carry else []
                current_count = len(carry)
            else:
                current = []
                current_count = 0
        current.append(sent)
        current_count += wc
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]
