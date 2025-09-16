# RAG OpenAI – Starter Modulaire

Starter modulaire pour construire un RAG simple et robuste sur des dossiers hétérogènes.

## 1) Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Édite .env et renseigne OPENAI_API_KEY
```

### Dépendances système pour OCR

* macOS: `brew install tesseract poppler`
* Debian/Ubuntu: `sudo apt-get install tesseract-ocr poppler-utils`

## 2) Ingestion

```bash
python scripts/ingest.py --root /chemin/vers/dossiers
```

## 3) API

```bash
uvicorn main:app --reload --port 8000
```

### Endpoints

* `GET /health`
* `POST /ingest` `{ "root_dir": "/path" }`
* `POST /query` `{ "question": "...", "k": 20, "top_n": 5 }`

## 4) Notes & bonnes pratiques

* **Embeddings**: `text-embedding-3-small` par défaut (coût/latence). Passe à `-large` si rappel insuffisant.
* **Chunking**: 1000/150 (tokens approximés par mots) → ajuste selon ton corpus.
* **Persistance**: FAISS (`.rag_index/vectors.faiss`) + JSONL aligné (`.rag_index/store.jsonl`).
* **Citations**: chaque chunk inclut `source` + `page/slide` si dispo.
* **Sécurité**: évite d’ingérer `.env`/clés/archives sensibles.
* **Améliorations faciles**: rerank cross-encoder, résumé de documents, déduplication par hash, filtres de type MIME.


## ✅ Test rapide (sans API)
1) Place quelques PDF/MD/CSV/images dans un dossier de test
2) `python scripts/ingest.py --root ./data`
3) Lance l’API puis poste une question sur `/query`

## 🔧 Idées d’extensions (si tu as un peu de temps)
- **Rerank**: HuggingFace cross-encoder multilingue après kNN.
- **Captioning images**: BLIP/Florence pour enrichir au-delà de l’OCR.
- **Watcher**: détection fichiers nouveaux/modifiés via hash + mtime.
- **Auth**: clé simple sur l’API FastAPI.
