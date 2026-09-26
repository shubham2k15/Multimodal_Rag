"""
Chroma wrapper. A single collection holds all chunk types (text, table,
image) since they all share one embedding space via gemini-embedding-2.
Embeddings are computed ourselves (via GeminiMultimodalEmbedder) and passed
in directly, rather than using Chroma's built-in embedding_function, so we
can control exactly which model/call is used per content type.
"""
import chromadb

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        ids = [c["chunk_id"] for c in chunks]
        documents = [self._document_text(c) for c in chunks]
        metadatas = [self._metadata(c) for c in chunks]

        print(f"Inserting {len(chunks)} chunks into Chroma collection '{self.collection.name}'...")
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return self._results_to_chunks(results)

    def get_all_chunks(self) -> list[dict]:
        """Used to build the BM25 index — pulls every stored chunk's text + metadata."""
        results = self.collection.get()
        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            chunks.append({
                "chunk_id": chunk_id,
                "searchable_text": results["documents"][i],
                **meta,
            })
        return chunks

    @staticmethod
    def _document_text(chunk: dict) -> str:
        """What BM25/keyword search and the stored 'document' field see."""
        if chunk["content_type"] == "image":
            return chunk.get("display_caption") or f"[image on page {chunk.get('page_number')}]"
        return chunk.get("searchable_text", "")

    @staticmethod
    def _metadata(chunk: dict) -> dict:
        meta = {
            "document_id": chunk.get("document_id"),
            "page_number": chunk.get("page_number") or 0,
            "section": chunk.get("section") or "",
            "content_type": chunk["content_type"],
        }
        if chunk["content_type"] == "table":
            meta["original_text"] = chunk["original_text"]
        elif chunk["content_type"] == "image":
            meta["image_path"] = chunk["image_path"]
        else:
            meta["original_text"] = chunk["original_text"]
        return meta

    @staticmethod
    def _results_to_chunks(results: dict) -> list[dict]:
        chunks = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for i, chunk_id in enumerate(ids):
            chunks.append({
                "chunk_id": chunk_id,
                "searchable_text": documents[i],
                **metadatas[i],
            })
        return chunks
