"""
Central configuration for the multimodal RAG pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- GCP Project Info ---
GCP_PROJECT = "project-6604d599-c74f-4139-a00"
GCP_LOCATION = "us-central1"

# --- Models ---
# Single unified embedding model for text, tables, and images.
# NOTE: "gemini-embedding-2" may be labeled "preview" depending on when you read this —
# check current availability at https://ai.google.dev/gemini-api/docs/models before
# committing production traffic to it. If unavailable, fall back to a two-model setup
# (gemini-embedding-001 for text/tables + a separate multimodal embedding call for images).
EMBEDDING_MODEL = "multimodalembedding"
EMBEDDING_OUTPUT_DIM = 1408  # Default dimensionality for multimodalembedding

GENERATION_MODEL = "gemini-2.5-flash"

# --- Chunking ---
MAX_TEXT_CHUNK_CHARS = 1000

# --- Retrieval ---
HYBRID_TOP_K = 20          # candidates pulled from vector + keyword search before reranking
RERANK_TOP_N = 5           # final number of chunks sent to the generation model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local, free, no API key needed

# --- Storage paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
IMAGE_STORE_DIR = os.path.join(DATA_DIR, "content_store", "images")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "multimodal_rag"

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
