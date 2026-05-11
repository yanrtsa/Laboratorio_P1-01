"""Módulo de reranking via Cross-Encoder.

O Cross-Encoder avalia cada par (query, documento) de forma conjunta,
produzindo um score de relevância mais preciso que o bi-encoder isolado,
ao custo de maior latência (O(n) inferências vs. O(1) do bi-encoder).
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, TOP_K_RERANK


class Reranker:
    """Reranking de documentos usando cross-encoder de alta precisão."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        print(f"[RERANKER] Carregando cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name, max_length=512)
        print("[RERANKER] Cross-encoder pronto.")

    def rerank(
        self,
        original_query: str,
        candidates: list[dict[str, Any]],
        top_k: int = TOP_K_RERANK,
    ) -> list[dict[str, Any]]:
        """Classifica candidatos por relevância usando a query original.

        O cross-encoder recebe (query, documento) como par e calcula um score
        de relevância profundo — diferente do bi-encoder que compara embeddings
        independentes, aqui há atenção cruzada entre query e documento.

        Args:
            original_query: Query original do usuário (linguagem natural).
            candidates: Lista de dicts retornados pelo retrieval.
            top_k: Número de resultados finais.

        Returns:
            Lista reordenada com os top_k documentos mais relevantes.
        """
        pairs = [(original_query, doc["content"]) for doc in candidates]
        scores = self._model.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top_results = reranked[:top_k]

        print(f"\n{'-' * 60}")
        print(f"[RERANK] Top-{top_k} apos Cross-Encoder:")
        print(f"{'-' * 60}")
        for rank, doc in enumerate(top_results, start=1):
            print(
                f"  {rank}. score={doc['rerank_score']:.4f} | "
                f"retrieval={doc['score']:.4f} | {doc['title']}"
            )
        print(f"{'-' * 60}")

        return top_results
