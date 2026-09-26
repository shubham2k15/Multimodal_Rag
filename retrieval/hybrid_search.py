"""
Hybrid retrieval: vector search (Chroma, via gemini-embedding-2) merged with
BM25 keyword search, combined via Reciprocal Rank Fusion (RRF).

Chroma has no native BM25, so we build a local BM25 index in memory over
every stored chunk's text at startup. This catches exact tokens (model
names, units, "Table 3", etc.) that dense embeddings can under-weight.
"""
from rank_bm25 import BM25Okapi

from config import HYBRID_TOP_K


class HybridRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self._build_bm25_index()

    def _build_bm25_index(self):
        all_chunks = self.vector_store.get_all_chunks()
        self.chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
        # Image chunks have little/no useful text for BM25 — they still get an
        # entry (their caption or a placeholder) so ids line up, but will rarely
        # score highly on keyword search, which is expected and fine: they're
        # primarily found via vector search instead.
        corpus = [c["searchable_text"].split() for c in all_chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None
        self.bm25_ids = [c["chunk_id"] for c in all_chunks]

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
        query_embedding = self.embedder.embed_text(query)

        vector_hits = self.vector_store.query(query_embedding, top_k=top_k)
        vector_ids = [h["chunk_id"] for h in vector_hits]
        # Ensure vector-hit chunks are available even if the BM25 index was
        # built before this chunk existed (e.g. incremental ingestion).
        for h in vector_hits:
            self.chunks_by_id.setdefault(h["chunk_id"], h)

        keyword_ids = []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query.split())
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
            keyword_ids = [self.bm25_ids[i] for i in top_idx]

        fused_ids = self._rrf_merge([vector_ids, keyword_ids])
        return [self.chunks_by_id[cid] for cid in fused_ids[:top_k] if cid in self.chunks_by_id]

    @staticmethod
    def _rrf_merge(id_lists: list[list[str]], k: int = 60) -> list[str]:
        scores = {}
        for ids in id_lists:
            for rank, chunk_id in enumerate(ids):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
