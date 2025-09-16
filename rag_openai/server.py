from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any

from .index import RAGIndex
from .loaders import ingest_directory

app = FastAPI(title="RAG OpenAI Starter", version="0.1.0")
_index: RAGIndex | None = None


def get_index() -> RAGIndex:
    global _index
    if _index is None:
        _index = RAGIndex()
    return _index


class IngestReq(BaseModel):
    root_dir: str


class QueryReq(BaseModel):
    question: str
    k: int = 20
    top_n: int = 5
    use_mmr: bool = False
    mmr_lambda: float = 0.5


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestReq):
    root = Path(req.root_dir)
    if not root.exists():
        return {"error": f"Path not found: {root}"}
    docs = ingest_directory(root)
    res = get_index().add_documents(docs)
    return {"files_ingested": len(docs), **res}


from .rag import answer_with_openai

@app.post("/query")
async def query(req: QueryReq):
    hits = get_index().search(req.question, k=req.k, use_mmr=req.use_mmr, top_n=req.top_n, lambda_param=req.mmr_lambda)
    top = hits[: max(1, req.top_n)]
    answer = answer_with_openai(req.question, top)
    return {"answer": answer, "sources": [h.get("metadata", {}) for h in top]}


# LangChain endpoint
class ChainReq(BaseModel):
    question: str
    k: int = 20
    top_n: int = 5

from .langchain_chain import answer_with_langchain

@app.post("/query_chain")
async def query_chain(req: ChainReq):
    res = answer_with_langchain(req.question, k=req.k, top_n=req.top_n)
    return res
