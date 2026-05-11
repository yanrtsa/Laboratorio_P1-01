"""Módulo de embeddings via Bi-Encoder (sentence-transformers)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Carrega e reutiliza o modelo de embedding em memória."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print(f"[EMBEDDINGS] Carregando modelo: {model_name}")
        self._model = SentenceTransformer(model_name)
        dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        print(f"[EMBEDDINGS] Modelo carregado. Dimensao: {dim}")

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Gera embeddings normalizados para uma lista de textos.

        Args:
            texts: Lista de strings a serem codificadas.
            normalize: Se True, normaliza os vetores para norma unitária (L2),
                       necessário para que distância coseno = 1 - produto interno.
            show_progress: Exibe barra de progresso para datasets grandes.
            batch_size: Tamanho do lote de inferência.

        Returns:
            Array numpy de forma (len(texts), dim_embedding).
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings  # type: ignore[return-value]

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Conveniência para codificar um único texto."""
        return self.encode([text], normalize=normalize)[0]

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()  # type: ignore[return-value]
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]
