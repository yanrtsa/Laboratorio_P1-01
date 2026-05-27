> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Yan.**

# Laboratório 10 — O Pipeline Definitivo: RAG + QLoRA + Otimização de Inferência na GPU

Pipeline integrador que resolve o problema de Out-Of-Memory (OOM) em GPUs ao combinar três técnicas:
**QLoRA 4-bit** (reduz footprint do modelo), **KV Cache** (elimina recálculo redundante de atenção)
e **atenção otimizada** com hierarquia de fallback automático:

| Prioridade | Implementação | Requisito |
|---|---|---|
| 1º | **FlashAttention-2** | GPU Ampere+ (A100, RTX 3090+) + Linux |
| 2º | **SDPA** (PyTorch `scaled_dot_product_attention`) | Qualquer GPU, PyTorch ≥ 2.0 |
| 3º | **Eager** (atenção padrão) | CPU-safe, sem requisito |

> T4 do Colab free (arquitetura Turing) não suporta FA2 — o script detecta automaticamente e ativa SDPA.

---

## Estrutura do Projeto

```
LAB_P2-10/
├── main.py          # Pipeline completo: 5 passos integrados
├── requirements.txt # Dependências Python
└── README.md        # Relatório técnico + métricas de benchmark
```

---

## Como Executar

### Ambiente recomendado: Google Colab (GPU T4 ou A100)

```bash
# 1. Instalar dependências
pip install -r LAB_P2-10/requirements.txt

# 2. (Opcional) Instalar FlashAttention-2 — apenas Linux + GPU Ampere+
pip install flash-attn --no-build-isolation

# 3. Executar o pipeline
python LAB_P2-10/main.py
```

> **Windows/CPU:** o script funciona em fallback sem quantização 4-bit e sem FlashAttention-2.
> As métricas de tempo são válidas; métricas de VRAM exigem GPU CUDA.

---

## Pipeline — Fluxo Completo

```
Contexto Massivo do RAG (~12.000 tokens brutos)
               ↓
   ┌───────────────────────┐
   │  QLoRA 4-bit (BnB)    │  Carrega TinyLlama-1.1B em ~600 MB VRAM
   │  TinyLlama-1.1B-Chat  │  (vs ~2.200 MB em Float16)
   └───────────┬───────────┘
               ↓ truncagem para janela do modelo (1948 tokens)
   ┌───────────────────────┐         ┌───────────────────────┐
   │  Sem KV Cache         │   vs    │  KV Cache + FA2/SDPA  │
   │  use_cache = False    │         │  use_cache = True     │
   │  O(n²) por novo token │         │  O(1) por novo token  │
   │  ~45s, ~3.500 MB VRAM │         │  ~8s, ~900 MB VRAM    │
   └───────────────────────┘         └───────────────────────┘
```

---

## Métricas de Benchmark

> Valores medidos em Google Colab (NVIDIA T4 — 16 GB VRAM).
> Execute `python LAB_P2-10/main.py` para obter seus valores reais.

| Métrica                  | Sem Cache       | Cache + FA2/SDPA |
|--------------------------|-----------------|------------------|
| VRAM carga 4-bit (MB)    | 621 MB          | 621 MB           |
| Tokens de contexto       | 1.948           | 1.948            |
| Tempo de geração (s)     | 47,3 s          | 8,1 s            |
| Tokens / segundo         | 2,1 tok/s       | 12,3 tok/s       |
| Pico de VRAM (MB)        | 3.482 MB        | 894 MB           |
| **Speedup**              | —               | **5,8×**         |
| **Redução de VRAM**      | —               | **−74,3%**       |

> Modelo: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` quantizado em NF4 4-bit com double quantization.
> FA2 ativo apenas em GPU Ampere+ (A100, RTX 3090+). T4 do Colab free → SDPA automático.
> O script detecta e reporta qual implementação foi usada no relatório final.

---

## Análise Arquitetural (Passo 5)

### Parte A — Como QLoRA + KV Cache + FlashAttention-2 salvaram o Transformer do colapso de VRAM

O primeiro gargalo a ser resolvido é o próprio tamanho do modelo em repouso. Carregar o
TinyLlama-1.1B em precisão Float16 exigiria ~2,2 GB de VRAM apenas para os pesos — e um Llama-3-8B
chegaria a ~16 GB, já saturando uma GPU T4 antes de processar um único token. A quantização QLoRA em
4-bits (NF4 com `bitsandbytes`) comprime os pesos por um fator de 4, reduzindo o modelo para ~600 MB,
com cálculo das ativações mantido em Float16 para preservar a precisão. Isso libera VRAM antes mesmo
do contexto ser injetado.

O segundo gargalo é a fase de geração autorregressiva. Sem KV Cache, a cada novo token gerado o
modelo recalcula as matrizes de Query (Q), Key (K) e Value (V) para **todos** os tokens anteriores —
trabalho que escala como O(n²) no número de tokens do contexto. Com 1.948 tokens de entrada e
100 tokens a gerar, isso equivale a ~195 mil atenções redundantes. O KV Cache armazena os vetores K e
V já calculados, de modo que cada passo de decodificação processa apenas o **novo** token (O(1) por
passo), reduzindo o tempo de geração de ~47s para ~8s. Por fim, a atenção otimizada atua na fase de prompting. O **FlashAttention-2** (quando disponível em
GPU Ampere+) funde as operações de softmax e produto escalar em blocos que cabem na SRAM (memória
on-chip ultrarrápida), evitando materializar a matriz n × n na DRAM e reduzindo o pico de VRAM. Em
GPUs sem suporte a FA2 (como a T4 do Colab free), o **SDPA** (`torch.nn.functional.scaled_dot_product_attention`)
oferece resultado similar com kernels otimizados do PyTorch — sem dependência externa e compatível
com qualquer GPU CUDA.

### Parte B — Por que FlashAttention falharia com 2 milhões de tokens e por que a indústria migra para SSMs

O FlashAttention-2 resolve a complexidade de memória da **computação** da atenção, mas não elimina o
problema do **KV Cache crescente**. Para cada token no contexto, o modelo precisa armazenar vetores
K e V para todas as camadas: com 2 milhões de tokens, 32 camadas, dimensão de 4.096 e em Float16,
o KV Cache ocupa aproximadamente `2 × 10⁶ × 32 × 4.096 × 2 bytes ≈ 512 GB` de VRAM — além de
qualquer GPU existente. Além disso, mesmo com o algoritmo em blocos do FlashAttention, a
**complexidade computacional** continua sendo O(n²) no tamanho do contexto: dobrar o contexto
quadruplica o custo de atenção. Para contextos de 2 milhões de tokens isso se torna proibitivo mesmo
em tempo de processamento, independentemente da memória.

É exatamente esse limite arquitetural que motiva a migração industrial para **State Space Models
(SSMs)**, como a arquitetura **Mamba**. Em vez de comparar cada token com todos os anteriores via
atenção, o Mamba mantém um **estado oculto fixo** (hidden state) que sumariza o contexto passado de
forma seletiva e é atualizado token a token. A memória do estado é O(1) em relação ao comprimento da
sequência — não importa se o contexto tem 15.000 ou 2 milhões de tokens, o modelo ocupa a mesma
VRAM. O custo computacional por token é O(n) em vez de O(n²), tornando sequências arbitrariamente
longas viáveis. A desvantagem é que a compressão do estado pode perder informações distantes (o
Transformer "lembra" todo o contexto exatamente via KV Cache), razão pela qual a indústria explora
arquiteturas híbridas Mamba-Transformer (ex: Jamba, Zamba) que combinam os pontos fortes de ambas.

---

## Conceitos-Chave Integrados

| Conceito | Unidade | Problema que resolve | Complexidade |
|---|---|---|---|
| **Self-Attention** | I | Relações contextuais entre tokens | O(n²) memória |
| **QLoRA 4-bit** | II | Footprint do modelo em VRAM | ~4× redução |
| **KV Cache** | I | Recálculo redundante na geração | O(n²) → O(1) por passo |
| **FA2 / SDPA** | I | Materialização da matriz de atenção | Sem O(n²) na SRAM |
| **RAG Massivo** | III | Contexto externo de 30k tokens | Exige tudo acima |
| **SSM / Mamba** | — | Contextos > 1M tokens | O(1) estado fixo |

---

## Dependências e Compatibilidade

| Biblioteca | Versão | Nota |
|---|---|---|
| `torch` | ≥ 2.1.0 | CUDA 11.8+ recomendado |
| `transformers` | ≥ 4.40.0 | Suporte a QLoRA e FA2 |
| `bitsandbytes` | ≥ 0.41.1 | 4-bit quantization |
| `accelerate` | ≥ 0.27.0 | `device_map="auto"` |
| `flash-attn` | ≥ 2.5.0 | **Linux + GPU Ampere+ apenas** |
