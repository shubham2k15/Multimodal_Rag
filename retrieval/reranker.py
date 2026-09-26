"""
Cross-encoder reranking over the hybrid-search candidates. Runs locally via
sentence-transformers (no additional API key needed) — scores each
(query, chunk_text) pair jointly, which is far more precise than comparing
independent embeddings, and is cheap enough to run only on the ~20 candidates
that already survived hybrid search.

Note: for image chunks, reranking scores the caption/placeholder text, not
the image itself — the cross-encoder model is text-only. This is a known
limitation; image relevance is primarily established by the vector search
stage (gemini-embedding-2 comparing the query directly against the image
embedding), and reranking mainly refines ordering among text/table chunks.
"""
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANK_TOP_N


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, c["searchable_text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: -x[1])
        return [chunk for chunk, _ in scored[:top_n]]
