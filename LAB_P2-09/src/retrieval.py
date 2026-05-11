"""Módulo de retrieval — busca Top-K no índice HNSW via bi-encoder."""

from __future__ import annotations

from typing import Any

from src.config import TOP_K_RETRIEVAL
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def retrieve(
    query_text: str,
    embedder: EmbeddingModel,
    store: VectorStore,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict[str, Any]]:
    """Executa busca por similaridade coseno no índice HNSW.

    Args:
        query_text: Texto de consulta (após HyDE, é o documento hipotético).
        embedder: Instância do modelo de embedding.
        store: Banco vetorial com índice HNSW.
        top_k: Número de resultados a recuperar.

    Returns:
        Lista ordenada por score descendente de dicts com id, title, content, score.
    """
    query_embedding = embedder.encode_single(query_text, normalize=True).tolist()
    hits = store.query(query_embedding, top_k=top_k)

    print(f"\n{'-' * 60}")
    print(f"[RETRIEVAL] Top-{top_k} documentos recuperados:")
    print(f"{'-' * 60}")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank:2d}. score={hit['score']:.4f} | {hit['title']}")
    print(f"{'-' * 60}")

    return hits
