# Multimodal RAG (Unstructured.io + gemini-embedding-2 + Gemini 2.5 Flash)

## Architecture

```
PDF
 -> Unstructured.io (hi_res parsing: text, tables as HTML, images extracted to disk)
 -> classify + section-tag elements
 -> chunk (text grouped by section; tables/images kept whole)
 -> gemini-embedding-2 embeds every chunk directly (text, table HTML, and raw images
    all mapped into one shared vector space)
 -> Chroma (single collection, persisted locally)
 -> query: hybrid search (Chroma vector search + BM25 keyword search, merged via
    Reciprocal Rank Fusion)
 -> rerank top ~20 candidates with a local cross-encoder -> top 5
 -> fetch ORIGINAL content for the top chunks (real table HTML, real image files)
 -> Gemini 2.5 Flash generates the final answer from question + real content
```

## Setup

```bash
pip install -r requirements.txt
```

System dependencies for Unstructured's PDF parsing (macOS example):

```bash
brew install poppler tesseract libmagic
```

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

Place a PDF in `data/raw_pdfs/document.pdf`, or pass a path directly.

## Run

```bash
python main.py                       # uses data/raw_pdfs/document.pdf
python main.py /path/to/your.pdf     # or specify a path
```

First run ingests the PDF (parses, embeds, stores in Chroma at
`data/content_store/` and `data/chroma_db/`). Subsequent runs skip ingestion
if the collection is already populated — delete `data/chroma_db/` to force
re-ingestion.

## Notes / things to check before production use

- **`gemini-embedding-2`** may be labeled "preview" — verify current
  availability/pricing at https://ai.google.dev/gemini-api/docs/models
  before relying on it long-term. If unavailable, `ingestion/embedder.py`
  is the only file that needs to change (swap in `gemini-embedding-001`
  for text/table chunks, and handle images separately).
- **Reranking** uses a local `sentence-transformers` cross-encoder so no
  extra API key is required. It scores text pairs only — image relevance
  ranking is primarily driven by the vector search stage, not reranking.
- **Chroma** persists locally under `data/chroma_db/`. For multi-document
  or production use, revisit access control, incremental ingestion, and
  duplicate detection (see the design doc) before scaling this up.
- `generate_captions=True` in `ingestion/pipeline.ingest_pdf(...)` will
  additionally ask Gemini 2.5 Flash to write a human-readable caption per
  image, purely for logging/display — it is not used for embedding or
  search.
