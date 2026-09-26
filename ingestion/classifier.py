"""
Walks parsed elements in document order, tracks the current section (last
seen Title), tags every element with content_type/page_number/section, and
attaches FigureCaption text to the nearest preceding Image/Figure element.
"""

CATEGORY_TO_CONTENT_TYPE = {
    "Table": "table",
    "Image": "image",
    "Figure": "image",
    "FigureCaption": "caption",
}


def classify_and_tag(elements, document_id: str) -> list[dict]:
    tagged = []
    current_section = None

    for el in elements:
        if el.category == "Title":
            current_section = el.text
            continue  # titles become section labels, not standalone chunks

        content_type = CATEGORY_TO_CONTENT_TYPE.get(el.category, "text")

        tagged.append({
            "document_id": document_id,
            "page_number": getattr(el.metadata, "page_number", None),
            "section": current_section,
            "content_type": content_type,
            "element": el,
        })

    # Attach captions to the nearest preceding image/figure element
    for i, item in enumerate(tagged):
        if item["content_type"] == "caption" and i > 0 and tagged[i - 1]["content_type"] == "image":
            tagged[i - 1]["caption"] = item["element"].text

    return [t for t in tagged if t["content_type"] != "caption"]
