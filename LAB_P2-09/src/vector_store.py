"""Banco vetorial com índice HNSW via ChromaDB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    HNSW_SEARCH_EF,
    HNSW_SPACE,
    MEDICAL_DATA_PATH,
    TOP_K_RETRIEVAL,
)
from src.embeddings import EmbeddingModel


class VectorStore:
    """Gerencia o índice HNSW persistente via ChromaDB.

    Hiperparâmetros HNSW configurados explicitamente nos metadados da coleção:
      - hnsw:M              → número de conexões por nó no grafo
      - hnsw:construction_ef → tamanho da lista dinâmica de candidatos na indexação
      - hnsw:search_ef      → tamanho da lista dinâmica de candidatos na busca
      - hnsw:space          → função de distância (cosine, l2, ip)
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedder = embedding_model
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # Cria ou recupera coleção com hiperparâmetros HNSW explícitos
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:M": HNSW_M,
                "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
                "hnsw:search_ef": HNSW_SEARCH_EF,
                "hnsw:space": HNSW_SPACE,
            },
        )
        print(
            f"[VECTOR STORE] Coleção '{COLLECTION_NAME}' | "
            f"M={HNSW_M} | ef_construction={HNSW_EF_CONSTRUCTION} | "
            f"ef_search={HNSW_SEARCH_EF} | space={HNSW_SPACE}"
        )

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def index_documents(self, data_path: Path = MEDICAL_DATA_PATH) -> None:
        """Lê o JSON de manuais médicos, gera embeddings e indexa no HNSW."""
        with open(data_path, "r", encoding="utf-8") as f:
            documents: list[dict[str, Any]] = json.load(f)

        print(f"[VECTOR STORE] Indexando {len(documents)} documentos...")

        ids, texts, metadatas = [], [], []
        for doc in documents:
            ids.append(str(doc["id"]))
            texts.append(doc["content"])
            metadatas.append({"title": doc["title"], "doc_id": doc["id"]})

        embeddings = self._embedder.encode(
            texts, normalize=True, show_progress=True
        ).tolist()

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[VECTOR STORE] {len(documents)} documentos indexados com sucesso.")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """Consulta o índice HNSW e retorna os top-k documentos mais similares.

        Args:
            query_embedding: Vetor de query já normalizado.
            top_k: Número de resultados a retornar.

        Returns:
            Lista de dicts com 'id', 'title', 'content' e 'score'.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB retorna distância coseno (1 - similaridade). Converter para score.
            distance = results["distances"][0][i]
            score = 1.0 - distance
            hits.append(
                {
                    "id": doc_id,
                    "title": results["metadatas"][0][i]["title"],
                    "content": results["documents"][0][i],
                    "score": round(score, 4),
                }
            )

        return hits
