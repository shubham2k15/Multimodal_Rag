"""
Builds the multimodal context from reranked chunks (fetching ORIGINAL
content — real table HTML, real image files — not the embedding proxy) and
generates the final answer with Gemini 2.5 Flash.
"""
from PIL import Image
from google import genai

from config import GENERATION_MODEL, GCP_PROJECT, GCP_LOCATION

SYSTEM_PROMPT = (
    "You are an assistant for question-answering over technical documents. "
    "Use ONLY the retrieved content provided below — text, tables, and figures. "
    "Use exact values from tables when asked about numbers, rows, or comparisons. "
    "Describe what you see directly when a figure or diagram is provided. "
    "Cite the page number(s) you used for your answer. "
    "If the answer is not present in the provided content, say so explicitly — "
    "do not guess or use outside knowledge."
)


def build_context_blocks(chunks: list[dict]) -> list:
    """
    Returns a list of content parts (str or PIL.Image) ready to pass to the
    Gemini API's `contents` argument alongside the question.
    """
    blocks = []
    for chunk in chunks:
        header = f"[Page {chunk.get('page_number')}, Section: {chunk.get('section') or 'N/A'}]"

        if chunk["content_type"] == "table":
            blocks.append(f"{header}\nTable:\n{chunk['original_text']}")

        elif chunk["content_type"] == "image":
            blocks.append(f"{header}\nFigure:")
            blocks.append(Image.open(chunk["image_path"]))

        else:  # text
            blocks.append(f"{header}\n{chunk['original_text']}")

    return blocks


class GeminiRAG:
    def __init__(self, model_name: str = GENERATION_MODEL, temperature: float = 0.2):
        self.client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        self.model_name = model_name
        self.temperature = temperature

    def answer_question(self, question: str, context_blocks: list) -> str:
        contents = [SYSTEM_PROMPT, f"Question: {question}"] + context_blocks

        print(f"Sending request to {self.model_name}...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={"temperature": self.temperature},
        )
        return response.text
