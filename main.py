"""
Entry point: ingests a PDF (if not already indexed) and runs an interactive
question loop using hybrid search + reranking + Gemini 2.5 Flash generation.
"""
import os
import sys

from config import RAW_PDF_DIR
from database.vector_store import VectorStore
from ingestion.embedder import GeminiMultimodalEmbedder
from ingestion.pipeline import ingest_pdf
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker
from generation.multimodal_llm import GeminiRAG, build_context_blocks


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAW_PDF_DIR, "document.pdf")
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]

    embedder = GeminiMultimodalEmbedder()
    vector_store = VectorStore()

    if not vector_store.is_populated():
        if not os.path.exists(pdf_path):
            print(f"No PDF found at {pdf_path}. Place a PDF there or pass a path as an argument.")
            sys.exit(1)
        print("\nVector store empty — running ingestion pipeline...")
        ingest_pdf(pdf_path, document_id, embedder, vector_store)
    else:
        print("\nVector store already populated — skipping ingestion.")

    retriever = HybridRetriever(vector_store, embedder)
    reranker = Reranker()
    rag = GeminiRAG()

    print("\n" + "=" * 50)
    print("Multimodal RAG system ready!")
    print("Type your question below. Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("Running hybrid search...")
            candidates = retriever.search(question)

            print(f"Reranking {len(candidates)} candidates...")
            top_chunks = reranker.rerank(question, candidates)

            if not top_chunks:
                print("No relevant content found.")
                continue

            print("Building multimodal context and generating answer...")
            context_blocks = build_context_blocks(top_chunks)
            answer = rag.answer_question(question, context_blocks)

            print("\n" + "=" * 50)
            print("Answer:")
            print(answer)
            print("=" * 50)
            print("\nSources used:")
            for c in top_chunks:
                print(f"  - Page {c.get('page_number')} ({c['content_type']}), section: {c.get('section')}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
