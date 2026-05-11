"""
LAB_P2-09 — Pipeline RAG Avançado para Manuais Médicos

Fluxo:
    User Query → HyDE → Embedding → HNSW Retrieval → Cross-Encoder Rerank → Top-3
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Força UTF-8 no stdout para compatibilidade com Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Garante que o diretório LAB_P2-09 esteja no path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MEDICAL_DATA_PATH, TOP_K_RERANK, TOP_K_RETRIEVAL
from src.embeddings import EmbeddingModel
from src.hyde import generate_hypothetical_document
from src.reranker import Reranker
from src.retrieval import retrieve
from src.vector_store import VectorStore


def build_pipeline() -> tuple[EmbeddingModel, VectorStore, Reranker]:
    """Inicializa e retorna todos os componentes do pipeline RAG."""
    embedder = EmbeddingModel()
    store = VectorStore(embedder)
    reranker = Reranker()

    if store.is_empty():
        print("\n[PIPELINE] Índice vazio — indexando documentos médicos...")
        store.index_documents(MEDICAL_DATA_PATH)
    else:
        print("\n[PIPELINE] Índice HNSW já persistido. Reutilizando.")

    return embedder, store, reranker


def run_query(
    query: str,
    embedder: EmbeddingModel,
    store: VectorStore,
    reranker: Reranker,
) -> list[dict]:
    """Executa o pipeline RAG completo para uma query.

    Etapas:
        1. HyDE: transforma query coloquial em documento hipotético técnico.
        2. Embedding: gera vetor do documento hipotético.
        3. HNSW Retrieval: recupera top-K candidatos por similaridade coseno.
        4. Cross-Encoder Rerank: rerankeia os candidatos com o modelo de precisão.

    Args:
        query: Pergunta/relato do usuário em linguagem natural.
        embedder: Modelo de embedding (bi-encoder).
        store: Banco vetorial HNSW persistente.
        reranker: Cross-encoder para reranking.

    Returns:
        Lista dos top-K documentos finais com scores.
    """
    print(f"\n{'=' * 60}")
    print(f"[QUERY] {query}")
    print(f"{'=' * 60}")

    # Etapa 1 — HyDE
    print("\n[HyDE] Gerando documento hipotético...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"\n[HyDE] Documento hipotetico gerado:\n  > {hypothetical_doc[:200]}...")

    # Etapa 2 + 3 — Embedding + HNSW Retrieval
    print(f"\n[RETRIEVAL] Buscando top-{TOP_K_RETRIEVAL} via HNSW...")
    candidates = retrieve(hypothetical_doc, embedder, store, top_k=TOP_K_RETRIEVAL)

    # Etapa 4 — Cross-Encoder Rerank
    print(f"\n[RERANK] Rerankeia top-{TOP_K_RETRIEVAL} → seleciona top-{TOP_K_RERANK}...")
    final_results = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)

    return final_results


def display_final_results(results: list[dict]) -> None:
    """Exibe os resultados finais de forma clara."""
    print(f"\n{'=' * 60}")
    print("  RESULTADO FINAL -- TOP-3 DOCUMENTOS MAIS RELEVANTES")
    print(f"{'=' * 60}")
    for rank, doc in enumerate(results, start=1):
        print(f"\n  #{rank} -- {doc['title']}")
        print(f"       Cross-Encoder Score : {doc['rerank_score']:.4f}")
        print(f"       Bi-Encoder Score    : {doc['score']:.4f}")
        snippet = doc["content"][:300].replace("\n", " ")
        print(f"       Trecho: {snippet}...")
    print(f"\n{'=' * 60}\n")


def main() -> None:
    """Ponto de entrada principal — demonstra o pipeline com queries médicas."""
    print("\n" + "#" * 60)
    print("  LAB P2-09 -- RAG Avancado para Manuais Medicos")
    print("  HNSW + HyDE + Bi-Encoder + Cross-Encoder")
    print("#" * 60)

    embedder, store, reranker = build_pipeline()

    # Queries de demonstração — mistura de coloquial e técnico
    demo_queries = [
        "dor de cabeça latejante e luz incomodando",
        "coração batendo rápido e inchaço nas pernas",
        "formigamento nos pés e açúcar alto no sangue",
        "tremor nas mãos e rigidez ao acordar",
        "falta de ar com chiado no peito",
    ]

    for query in demo_queries:
        results = run_query(query, embedder, store, reranker)
        display_final_results(results)

    # Modo interativo opcional
    print("\n" + "-" * 60)
    print("  Modo interativo -- digite sua query medica (ou 'sair' para encerrar)")
    print("-" * 60)
    while True:
        try:
            user_input = input("\n  Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() in {"sair", "exit", "quit"}:
            break

        results = run_query(user_input, embedder, store, reranker)
        display_final_results(results)

    print("\n[PIPELINE] Encerrado.\n")


if __name__ == "__main__":
    main()
