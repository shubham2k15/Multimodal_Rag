"""
Tables are kept as structured HTML (from Unstructured's table-structure
inference) and embedded directly with gemini-embedding-2 — no intermediate
text-description LLM call is needed, since the table's own HTML/text is
already what we embed AND what we show the final LLM.
"""


def process_table(tagged_item: dict) -> dict:
    table_html = tagged_item["element"].metadata.text_as_html or tagged_item["element"].text

    return {
        "document_id": tagged_item["document_id"],
        "page_number": tagged_item["page_number"],
        "section": tagged_item["section"],
        "content_type": "table",
        # searchable_text is what gets embedded — the real table itself.
        "searchable_text": table_html,
        # original_text is what gets shown to the final LLM — same content,
        # kept as a separate field for consistency with the text/image chunks
        # and so a future change (e.g. adding an LLM summary) only touches
        # searchable_text without affecting what generation sees.
        "original_text": table_html,
    }
