from __future__ import annotations
import argparse
from pathlib import Path
from rag_openai.loaders import ingest_directory
from rag_openai.index import RAGIndex


def main():
    parser = argparse.ArgumentParser(description="Ingestion de dossiers pour RAG")
    parser.add_argument("--root", required=True, help="Chemin vers le dossier racine")
    parser.add_argument("--verbose", action="store_true", help="Log détaillé par fichier")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Path not found: {root}")
        return

    print("[1/2] Ingestion…")
    docs = ingest_directory(root, verbose=args.verbose)
    print(f"  → {len(docs)} documents logiques (pages, slides, etc.)")

    print("[2/2] Indexation…")
    idx = RAGIndex()
    res = idx.add_documents(docs)
    print(f"  → Chunks ajoutés: {res['added']}")


if __name__ == "__main__":
    main()
