from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import json
import orjson
import numpy as np
import faiss
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
import uuid

from .config import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    INDEX_DIR,
    STORE_FILE,
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
    CHUNK_SENTENCE_AWARE,
    CHROMA_DIR,
)
from .utils import chunk_text


class OpenAIEmbedder:
    def __init__(self, model: str = OPENAI_EMBED_MODEL):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY manquant. Renseigne-le dans .env")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def embed(self, texts: Iterable[str], batch_size: int = 128) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1536), dtype="float32")
        batches: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            X = np.array([d.embedding for d in resp.data], dtype="float32")
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            batches.append(X)
        if not batches:
            return np.zeros((0, 1536), dtype="float32")
        return np.vstack(batches)


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    lambda_param: float = 0.5,
    top_n: int = 5,
) -> List[int]:
    # Maximal Marginal Relevance selection of indices
    if doc_vecs.size == 0:
        return []
    similarities = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()  # cosine if vecs normalized
    selected: List[int] = []
    candidates = list(range(len(similarities)))
    while candidates and len(selected) < top_n:
        if not selected:
            best = int(np.argmax(similarities[candidates]))
            selected.append(candidates.pop(best))
            continue
        selected_vecs = doc_vecs[selected]
        # diversity term: max similarity to any selected item
        div = np.max(selected_vecs @ doc_vecs[candidates].T, axis=0)
        mmr_scores = lambda_param * similarities[candidates] - (1 - lambda_param) * div
        pick = int(np.argmax(mmr_scores))
        selected.append(candidates.pop(pick))
    return selected


class FaissIndex:
    def __init__(self, dim: int = 1536):
        self.index: faiss.IndexFlatIP | None = None
        self.dim = dim

    def add(self, X: np.ndarray):
        if X.size == 0:
            return
        if self.index is None:
            self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

    def search(self, q: np.ndarray, k: int = 20):
        if self.index is None:
            return np.array([[0.0]]), np.array([[-1]])
        D, I = self.index.search(q, k)
        return D, I

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        faiss.write_index(self.index, str(path))

    def load(self, path: Path):
        if path.exists():
            self.index = faiss.read_index(str(path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)


class Store:
    """Stocke les chunks et métadonnées dans un JSONL aligné avec FAISS."""

    def __init__(self, file_path: Path = Path(STORE_FILE)):
        self.file_path = file_path
        self.records: List[Dict[str, Any]] = []
        if file_path.exists():
            with open(file_path, "rb") as f:
                for line in f:
                    self.records.append(orjson.loads(line))

    def append_many(self, items: List[Dict[str, Any]]):
        self.records.extend(items)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "ab") as f:
            for it in items:
                f.write(orjson.dumps(it))
                f.write(b"\n")

    def __len__(self):
        return len(self.records)

    def get(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class RAGIndex:
    def __init__(self, embed_model: str = OPENAI_EMBED_MODEL):
        self.embedder = OpenAIEmbedder(model=embed_model)
        self.faiss = FaissIndex()
        self.store = Store(Path(STORE_FILE))
        # Charger index FAISS s'il existe
        self.faiss.load(Path(INDEX_DIR) / "vectors.faiss")

    def add_documents(self, docs: List[Dict[str, Any]], chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
        chunks: List[Dict[str, Any]] = []
        for d in docs:
            for ch in chunk_text(d["text"], chunk_tokens, overlap, use_sentences=CHUNK_SENTENCE_AWARE):
                meta = dict(d["metadata"])  # copy
                meta["chunk_size"] = len(ch.split())
                chunks.append({"text": ch, "metadata": meta})
        if not chunks:
            return {"added": 0}
        X = self.embedder.embed([c["text"] for c in chunks])
        self.faiss.add(X)
        self.store.append_many(chunks)
        # Sauvegarder index
        self.faiss.save(Path(INDEX_DIR) / "vectors.faiss")
        return {"added": len(chunks)}

    def search(self, query: str, k: int = 20, use_mmr: bool = False, top_n: int | None = None, lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        qv = self.embedder.embed([query])
        D, I = self.faiss.search(qv, k)
        hits: List[Dict[str, Any]] = []
        valid_indices: List[int] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            if idx >= len(self.store.records):
                # Skip out-of-range ids (index/store misalignment)
                continue
            rec = self.store.get(int(idx))
            hits.append({**rec, "score": float(score)})
            valid_indices.append(int(idx))
        if use_mmr and hits:
            cand_vecs = self.embedder.embed([h["text"] for h in hits])
            selected_rel = mmr(qv[0], cand_vecs, lambda_param=lambda_param, top_n=top_n or min(5, len(hits)))
            hits = [hits[i] for i in selected_rel]
        return hits


class ChromaIndex:
    def __init__(self, embed_model: str = OPENAI_EMBED_MODEL, persist_dir: str = CHROMA_DIR):
        self.embedder = OpenAIEmbedder(model=embed_model)
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        self.client = Client(settings)
        self.collection = self.client.get_or_create_collection(name="rag_collection")

    def add_documents(self, docs, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
        chunks = []
        for d in docs:
            for ch in chunk_text(d["text"], max_tokens=chunk_tokens, overlap=overlap, use_sentences=True):
                meta = dict(d["metadata"])
                meta["chunk_size"] = len(ch.split())
                chunks.append({"text": ch, "metadata": meta})
        if not chunks: return {"added": 0}
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        self.client.persist()
        return {"added": len(chunks)}

    def search(self, query: str, k: int = 5):
        qvec = self.embedder.embed([query])
        results = self.collection.query(query_embeddings=qvec.tolist(), n_results=k)
        hits = []
        for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            hits.append({"text": doc, "metadata": meta})
        return hits
