"""
RAG Pipeline Configuration
All tunable constants in one place.
"""

from pathlib import Path

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
