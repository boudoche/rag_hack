from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import orjson
import numpy as np

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from .config import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    INDEX_DIR,
    STORE_FILE,
)


def _extract_faiss_index(store: FaissVectorStore) -> faiss.Index:
    # Try common attribute names across versions
    for attr in ("faiss_index", "index", "_faiss_index", "_index"):
        idx = getattr(store, attr, None)
        if idx is not None:
            return idx
    # Some versions keep it under client
    client = getattr(store, "client", None)
    if client is not None:
        return client
    raise AttributeError("Unable to access underlying FAISS index from FaissVectorStore")


def ingest_with_llamaindex(root_dir: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    # Load documents (LlamaIndex readers handle common formats)
    docs = SimpleDirectoryReader(input_dir=str(root), recursive=True).load_data()

    # Ensure FAISS index path
    index_path = Path(INDEX_DIR) / "vectors.faiss"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Create or load FAISS index
    faiss_index = faiss.read_index(str(index_path)) if index_path.exists() else faiss.IndexFlatIP(1536)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Embeddings
    embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _ = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)

    # Persist FAISS (robust to version differences)
    faiss_idx = _extract_faiss_index(vector_store)
    faiss.write_index(faiss_idx, str(index_path))

    # Append store.jsonl for parity (store raw text + minimal metadata)
    added = 0
    Path(STORE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_FILE, "ab") as f:
        for d in docs:
            meta = {"source": d.metadata.get("file_path", "?"), "type": d.metadata.get("file_type", "doc")}
            rec = {"text": d.text, "metadata": meta}
            f.write(orjson.dumps(rec))
            f.write(b"\n")
            added += 1

    return {"files_ingested": len(docs), "added": added}
