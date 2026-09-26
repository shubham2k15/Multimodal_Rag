"""
Full ingestion pipeline: PDF -> parsed elements -> classified/tagged ->
chunked (text) / finalized (table, image) -> embedded -> stored in Chroma.
"""
from ingestion.parser import parse_pdf
from ingestion.classifier import classify_and_tag
from ingestion.chunker import chunk_text_elements, finalize_chunk
from ingestion.table_processor import process_table
from ingestion.image_processor import process_image


def ingest_pdf(pdf_path: str, document_id: str, embedder, vector_store, generate_captions: bool = False):
    elements = parse_pdf(pdf_path)
    tagged = classify_and_tag(elements, document_id)

    text_chunks = chunk_text_elements([t for t in tagged if t["content_type"] == "text"])

    table_chunks = [
        finalize_chunk(process_table(t))
        for t in tagged if t["content_type"] == "table"
    ]

    image_chunks = [
        finalize_chunk(process_image(t, generate_caption=generate_captions))
        for t in tagged if t["content_type"] == "image"
    ]

    all_chunks = text_chunks + table_chunks + image_chunks
    print(f"Built {len(all_chunks)} chunks "
          f"({len(text_chunks)} text, {len(table_chunks)} table, {len(image_chunks)} image)")

    embeddings = []
    for i, chunk in enumerate(all_chunks):
        print(f"Embedding chunk {i + 1}/{len(all_chunks)} ({chunk['content_type']})...")
        embeddings.append(embedder.embed_chunk(chunk))

    vector_store.add_chunks(all_chunks, embeddings)
    print(f"Ingestion complete for {pdf_path}")
    return all_chunks
