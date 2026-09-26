"""
Images/figures are embedded directly with gemini-embedding-2 (no text
description required for the embedding itself, since the model handles
images natively). An optional Gemini 2.5 Flash caption can be generated
purely for human-readable metadata/logging — it is NOT used for embedding
or retrieval.
"""
from PIL import Image
from google import genai

from config import GENERATION_MODEL, GCP_PROJECT, GCP_LOCATION


def process_image(tagged_item: dict, generate_caption: bool = False) -> dict:
    image_path = tagged_item["element"].metadata.image_path
    caption = tagged_item.get("caption", "")

    display_caption = caption
    if generate_caption:
        display_caption = _generate_caption(image_path, caption)

    return {
        "document_id": tagged_item["document_id"],
        "page_number": tagged_item["page_number"],
        "section": tagged_item["section"],
        "content_type": "image",
        "image_path": image_path,          # used both for embedding and for final generation
        "display_caption": display_caption,  # human-readable only, not embedded
    }


def _generate_caption(image_path: str, existing_caption: str) -> str:
    """Optional: Flash-generated caption for logging/debugging only."""
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    img = Image.open(image_path)
    prompt = (
        "Briefly describe this figure/diagram in one sentence for a human-readable log "
        f"(this is NOT used for search). Existing caption if any: {existing_caption}"
    )
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=[prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Caption generation failed for {image_path}: {e}")
        return existing_caption
