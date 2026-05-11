"""Configuração centralizada do pipeline RAG médico."""

from pathlib import Path

# ── Diretórios ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHROMA_DIR = OUTPUTS_DIR / "chroma_db"

# ── Modelos ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── HNSW — Hiperparâmetros ────────────────────────────────────────────────────
# M: número de conexões bidirecionais por nó durante a construção do grafo.
#    Valores maiores → maior recall, maior consumo de memória (RAM ∝ M).
#    Recomendado: 8–64. Padrão: 16.
HNSW_M = 16

# ef_construction: tamanho da lista de candidatos dinâmicos durante a indexação.
#    Valores maiores → qualidade superior do grafo, indexação mais lenta.
#    Deve ser ≥ 2*M. Padrão: 200.
HNSW_EF_CONSTRUCTION = 200

# ef_search (search_ef): candidatos avaliados durante a busca.
#    Maior ef_search → maior recall, maior latência de busca.
HNSW_SEARCH_EF = 100

# Função de distância para similaridade semântica.
HNSW_SPACE = "cosine"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10   # Número de documentos recuperados pelo bi-encoder
TOP_K_RERANK = 3       # Número de documentos após reranking cross-encoder

# ── HyDE ─────────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"

# ── Dataset ───────────────────────────────────────────────────────────────────
MEDICAL_DATA_PATH = DATA_DIR / "medical_manuals.json"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "medical_rag"
