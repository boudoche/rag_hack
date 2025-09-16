from __future__ import annotations
import argparse
from rag_openai.llama_ingest import ingest_with_llamaindex


def main():
    parser = argparse.ArgumentParser(description="Ingestion via LlamaIndex -> FAISS")
    parser.add_argument("--root", required=True, help="Chemin vers le dossier racine")
    args = parser.parse_args()

    print("[LlamaIndex] Ingestion…")
    res = ingest_with_llamaindex(args.root)
    print(f"  → Files ingested: {res['files_ingested']}")
    print(f"  → Added records: {res['added']}")


if __name__ == "__main__":
    main()
