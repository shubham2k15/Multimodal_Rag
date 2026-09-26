"""
Single unified embedding model (multimodalembedding) for text, table HTML,
and raw images using the vertexai SDK — all mapped into the same vector space.
"""
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage

from config import EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM, GCP_PROJECT, GCP_LOCATION


class GeminiMultimodalEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, output_dim: int = EMBEDDING_OUTPUT_DIM):
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        self.model = MultiModalEmbeddingModel.from_pretrained(model_name)
        self.output_dim = output_dim

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(15))
    def embed_text(self, text: str) -> list[float]:
        """Embeds text (used for text chunks and table HTML alike)."""
        time.sleep(3)  # Throttle to avoid hitting QPM quota
        safe_text = text[:1000]
        embeddings = self.model.get_embeddings(contextual_text=safe_text)
        return embeddings.text_embedding

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(15))
    def embed_image(self, image_path: str) -> list[float]:
        """Embeds a raw image directly — no text description involved."""
        time.sleep(3)  # Throttle to avoid hitting QPM quota
        img = VertexImage.load_from_file(image_path)
        embeddings = self.model.get_embeddings(image=img)
        return embeddings.image_embedding

    def embed_chunk(self, chunk: dict) -> list[float]:
        """Dispatches to the right embed call based on content_type."""
        if chunk["content_type"] == "image":
            return self.embed_image(chunk["image_path"])
        return self.embed_text(chunk["searchable_text"])
