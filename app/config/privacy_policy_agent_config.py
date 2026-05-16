"""
RAG Pipeline Configuration
All tunable constants in one place.
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

privacy_policy_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # → AgenticBanking/
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma"

# ── Embedding ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-V2"

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", " ", ""]

# ── ChromaDB ───────────────────────────────────────────
COLLECTION_NAME = "pdf_documents"

# ── Retrieval ──────────────────────────────────────────
TOP_K = 5
SCORE_THRESHOLD = 0.4

