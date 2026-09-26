import os
import streamlit as st
from config import RAW_PDF_DIR
from database.vector_store import VectorStore
from ingestion.embedder import GeminiMultimodalEmbedder
from ingestion.pipeline import ingest_pdf
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker
from generation.multimodal_llm import GeminiRAG, build_context_blocks

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG Application")

# Initialize models (cached so they are not reloaded on every rerun)
@st.cache_resource
def get_components():
    embedder = GeminiMultimodalEmbedder()
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store, embedder)
    reranker = Reranker()
    rag = GeminiRAG()
    return embedder, vector_store, retriever, reranker, rag

embedder, vector_store, retriever, reranker, rag = get_components()

os.makedirs(RAW_PDF_DIR, exist_ok=True)

# File upload section
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join(RAW_PDF_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    document_id = os.path.splitext(uploaded_file.name)[0]
    
    if st.sidebar.button("Index Document"):
        with st.spinner("Indexing document... This may take a moment."):
            ingest_pdf(pdf_path, document_id, embedder, vector_store)
            st.sidebar.success("Document indexed successfully!")

# Question answering section
st.header("Ask a Question")
question = st.text_input("Enter your question based on the uploaded document:")

if st.button("Submit Question") and question:
    with st.spinner("Searching and generating answer..."):
        try:
            candidates = retriever.search(question)
            top_chunks = reranker.rerank(question, candidates)
            
            if not top_chunks:
                st.warning("No relevant content found in the document.")
            else:
                context_blocks = build_context_blocks(top_chunks)
                answer = rag.answer_question(question, context_blocks)
                
                st.markdown("### Answer")
                st.write(answer)
                
                st.markdown("### Sources used")
                for c in top_chunks:
                    st.write(f"- Page {c.get('page_number', 'N/A')} ({c.get('content_type', 'N/A')}), section: {c.get('section', 'N/A')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
