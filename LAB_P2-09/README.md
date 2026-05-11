# Laboratório 09 — RAG Avançado para Manuais Médicos

Pipeline de **Retrieval-Augmented Generation** com técnicas de ponta: índice **HNSW**, geração hipotética de documentos (**HyDE**), recuperação por **Bi-Encoder** e reranking por **Cross-Encoder**. O sistema funciona como buscador inteligente para manuais médicos privados, 100% offline.

---

## Estrutura do Projeto

```
LAB_P2-09/
├── data/
│   └── medical_manuals.json     # 25 fragmentos técnicos médicos gerados
├── src/
│   ├── __init__.py
│   ├── config.py                # Hiperparâmetros e configuração centralizada
│   ├── embeddings.py            # Bi-Encoder (all-MiniLM-L6-v2)
│   ├── vector_store.py          # Banco vetorial HNSW via ChromaDB
│   ├── hyde.py                  # HyDE com fallback offline obrigatório
│   ├── retrieval.py             # Retrieval Top-K com similaridade coseno
│   └── reranker.py              # Cross-Encoder reranking (ms-marco)
├── outputs/
│   └── chroma_db/               # Índice HNSW persistido automaticamente
├── main.py                      # Pipeline orquestrado + modo interativo
├── requirements.txt
└── README.md
```

---

## Como Executar

```bash
# 1. Instale as dependências
pip install -r LAB_P2-09/requirements.txt

# 2. Execute o pipeline (funciona 100% offline)
python LAB_P2-09/main.py

# 3. (Opcional) Com OpenAI para HyDE aprimorado
OPENAI_API_KEY=sk-... python LAB_P2-09/main.py
```

---

## Pipeline — Fluxo Completo

```
User Query (coloquial)
       ↓
   ┌─────────┐
   │  HyDE   │  Transforma query em documento clínico hipotético
   └────┬────┘
        ↓
   ┌──────────────┐
   │  Bi-Encoder  │  Gera embedding do documento hipotético
   │  (MiniLM)    │  sentence-transformers/all-MiniLM-L6-v2
   └──────┬───────┘
          ↓
   ┌──────────────┐
   │ HNSW Retrieval│  Busca Top-10 por similaridade coseno
   │  (ChromaDB)  │  M=16, ef_construction=200
   └──────┬───────┘
          ↓
   ┌──────────────┐
   │Cross-Encoder │  Reranking com atenção cruzada query-documento
   │ (ms-marco)   │  cross-encoder/ms-marco-MiniLM-L-6-v2
   └──────┬───────┘
          ↓
     Top-3 Finais
```

---

## Explicação Teórica

### RAG (Retrieval-Augmented Generation)

RAG é uma arquitetura que combina dois módulos: um **retriever** (busca documentos relevantes em uma base de conhecimento) e um **generator** (LLM que responde usando os documentos recuperados como contexto). O objetivo é reduzir alucinações e permitir que o modelo acesse conhecimento privado/atualizado sem fine-tuning.

### Embeddings e Bi-Encoder

Um **embedding** é uma representação vetorial densa de texto em um espaço semântico de alta dimensão, onde textos semanticamente similares ficam próximos geometricamente. O **Bi-Encoder** codifica query e documentos independentemente e compara os vetores via similaridade coseno — O(1) por par após indexação, ideal para recuperação em larga escala.

### HNSW (Hierarchical Navigable Small World)

HNSW é um algoritmo de busca aproximada de vizinhos mais próximos (ANN) baseado em grafos navegáveis hierárquicos. Ao contrário da busca exata (KNN bruta), que escala linearmente O(n·d), o HNSW alcança O(log n) na busca com recall próximo a 100%.

**Hiperparâmetros críticos:**

| Parâmetro | Valor | Impacto |
|-----------|-------|---------|
| `M` | 16 | Conexões por nó no grafo. Maior M → maior recall, maior RAM (RAM ∝ M × n × 4 bytes). |
| `ef_construction` | 200 | Candidatos durante indexação. Maior → grafo de qualidade superior, indexação mais lenta. Deve ser ≥ 2×M. |
| `ef_search` | 100 | Candidatos durante busca. Maior → maior recall, maior latência. Trade-off precisão/velocidade. |
| `space` | cosine | Função de distância. Coseno é ideal para embeddings normalizados (invariante a magnitude). |

**HNSW vs KNN Bruto:**

```
KNN Bruto:  O(n × d)   — escala linearmente, inviável com > 100k documentos
HNSW:       O(log n)   — escala logaritmicamente, recall > 95% em benchmarks (ANN-Benchmarks)
```

O grafo HNSW organiza nós em camadas hierárquicas: camadas superiores têm poucos nós com conexões de longo alcance (navegação rápida), camadas inferiores têm todos os nós com conexões locais (precisão). A busca desce as camadas com beam search guloso.

### HyDE (Hypothetical Document Embeddings)

HyDE resolve o problema de **mismatch semântico** entre queries coloquiais e documentos técnicos. Em vez de buscar pelo embedding da query diretamente, o sistema:

1. Usa um LLM para gerar um **documento hipotético** na linguagem do corpus (linguagem clínica).
2. Gera o embedding desse documento hipotético.
3. Usa esse embedding na busca vetorial.

O resultado é um vetor de busca muito mais próximo do espaço semântico dos documentos técnicos, melhorando o recall de forma significativa.

**Exemplo:**

```
Entrada  : "dor de cabeça latejante e luz incomodando"
Saída    : "Paciente apresenta cefaleia pulsátil associada à fotofobia.
            À anamnese, crises de 4–72 horas com náusea e fonofobia.
            Hipótese: enxaqueca conforme ICHD-3..."
```

**Fallback offline:** quando não há API key, o sistema usa mapeamento semântico local (60+ termos coloquiais → médicos) + templates clínicos por categoria, garantindo 100% de funcionamento offline.

### Cross-Encoder e Reranking

O **Cross-Encoder** recebe o par (query, documento) concatenado e aplica atenção cruzada completa entre os dois textos, produzindo um score de relevância muito mais preciso que o Bi-Encoder. O trade-off: é O(n) inferências (n = candidatos), por isso é usado apenas sobre o top-K do retrieval.

```
Bi-Encoder:    encode(query) ·  encode(doc) → similaridade aproximada, rápido
Cross-Encoder: encode(query + doc)          → score preciso, mais lento
```

O pipeline combina os dois: Bi-Encoder para recuperar candidatos eficientemente, Cross-Encoder para selecionar os mais relevantes com precisão máxima.

---

## Exemplo de Saída

```
════════════════════════════════════════════════════════════
[QUERY] dor de cabeça latejante e luz incomodando
════════════════════════════════════════════════════════════

[HyDE] Modo offline — usando mapeamento semântico local.
[HyDE] Documento hipotético gerado:
  → Paciente apresenta cefaleia pulsátil, fotofobia. À anamnese, crises
    recorrentes de cefaleia de moderada a forte intensidade...

────────────────────────────────────────────────────────────
[RETRIEVAL] Top-10 documentos recuperados:
────────────────────────────────────────────────────────────
   1. score=0.8312 | Enxaqueca (Migrânea)
   2. score=0.7245 | Meningite Bacteriana
   3. score=0.6891 | Doença de Alzheimer
   4. score=0.6712 | Neuropatia Periférica Diabética
   5. score=0.6534 | Epilepsia e Síndromes Epilépticas
   ...

────────────────────────────────────────────────────────────
[RERANK] Top-3 após Cross-Encoder:
────────────────────────────────────────────────────────────
   1. score=9.8241 | retrieval=0.8312 | Enxaqueca (Migrânea)
   2. score=4.1203 | retrieval=0.7245 | Meningite Bacteriana
   3. score=2.9845 | retrieval=0.6891 | Doença de Alzheimer

════════════════════════════════════════════════════════════
  RESULTADO FINAL — TOP-3 DOCUMENTOS MAIS RELEVANTES
════════════════════════════════════════════════════════════

  #1 — Enxaqueca (Migrânea)
       Cross-Encoder Score : 9.8241
       Bi-Encoder Score    : 0.8312
       Trecho: A enxaqueca é uma cefaleia primária caracterizada por dor
       pulsátil, geralmente unilateral, de intensidade moderada a grave...
```

---

## Requisitos

- Python 3.11+
- GPU (recomendado) ou CPU
- Sem necessidade de API externa (100% offline)

---

## Nota de Integridade

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Yan.
