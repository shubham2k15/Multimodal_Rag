"""
Element-level chunking:
- Text elements are grouped into section-bound chunks (never split across sections).
- Table and image elements are never split — each is already its own chunk.
"""
import uuid

from config import MAX_TEXT_CHUNK_CHARS


def chunk_text_elements(tagged_items: list[dict]) -> list[dict]:
    """Groups consecutive text elements sharing the same section into chunks."""
    chunks = []
    buffer, buffer_section, buffer_page, buffer_doc = "", None, None, None

    def flush():
        if buffer.strip():
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": buffer_doc,
                "page_number": buffer_page,
                "section": buffer_section,
                "parent_section": buffer_section,
                "content_type": "text",
                "searchable_text": buffer.strip(),
                "original_text": buffer.strip(),
            })

    for item in tagged_items:
        if item["content_type"] != "text":
            continue
        text = item["element"].text
        if buffer_section != item["section"] or len(buffer) + len(text) > MAX_TEXT_CHUNK_CHARS:
            flush()
            buffer, buffer_section, buffer_page, buffer_doc = "", item["section"], item["page_number"], item["document_id"]
        buffer += "\n" + text
        buffer_page = buffer_page or item["page_number"]
        buffer_doc = buffer_doc or item["document_id"]

    flush()
    return chunks


def finalize_chunk(processed: dict) -> dict:
    """Adds chunk_id/parent_section to an already-processed table or image dict."""
    processed["chunk_id"] = str(uuid.uuid4())
    processed["parent_section"] = processed.get("section")
    return processed
