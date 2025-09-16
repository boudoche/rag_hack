from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from .config import OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_CHAT_MODEL, INDEX_DIR, STORE_FILE, SYSTEM_PROMPT
import orjson


def load_faiss_and_store() -> tuple[FAISS, list[dict[str, Any]]]:
    index_path = Path(INDEX_DIR) / "vectors.faiss"
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found. Ingest first.")
    # Load FAISS Index and wrap into LangChain store
    faiss_index = faiss.read_index(str(index_path))
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)
    store = FAISS(embedding_function=embeddings, index=faiss_index, docstore=None, index_to_docstore_id=None)
    # Load our aligned store.jsonl to map indices to texts/metadata
    records: list[dict[str, Any]] = []
    with open(STORE_FILE, "rb") as f:
        for line in f:
            records.append(orjson.loads(line))
    return store, records


def build_context(records: list[dict[str, Any]], indices: List[int], limit_chars: int = 6000) -> str:
    parts: List[str] = []
    used = 0
    for i in indices:
        if i < 0 or i >= len(records):
            continue
        r = records[i]
        md = r.get("metadata", {})
        src = md.get("source", "?")
        loc = (
            f" p.{md.get('page')}" if md.get("page") is not None else (
                f" slide {md.get('slide')}" if md.get("slide") is not None else ""
            )
        )
        block = f"[SOURCE: {src}{loc}]\n{r['text']}\n"
        if used + len(block) > limit_chars:
            break
        used += len(block)
        parts.append(block)
    return "\n\n".join(parts)


def answer_with_langchain(question: str, k: int = 20, top_n: int = 5) -> dict[str, Any]:
    store, records = load_faiss_and_store()
    # Retrieve indices via FAISS
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)
    qvec = np.array([embeddings.embed_query(question)], dtype="float32")
    qvec /= np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
    D, I = store.index.search(qvec, k)
    idxs = [int(i) for i in I[0] if i >= 0][: max(1, top_n)]

    context = build_context(records, idxs)
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    msgs = [
        ("system", SYSTEM_PROMPT),
        ("user", f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations.")
    ]
    resp = llm.invoke(msgs)
    answer = resp.content
    sources = [records[i].get("metadata", {}) for i in idxs]
    return {"answer": answer, "sources": sources}
