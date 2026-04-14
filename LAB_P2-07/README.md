# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

Pipeline completo de fine-tuning do modelo **TinyLlama-1.1B** utilizando técnicas de eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA).

## Estrutura do Projeto

```
├── LAB_P2-07/
│   └── main-07.py        # Script principal (Passos 2, 3 e 4)
├── train.json             # Dataset de treino (90%)
├── test.json              # Dataset de teste (10%)
├── requirements.txt
└── README.md
```

## Como Executar

```bash
pip install -r requirements.txt
python LAB_P2-07/main-07.py
```

O adaptador treinado será salvo na pasta `./tinyllama-qlora-adapter`.

## Implementação

### Passo 1 — Geração do Dataset
Dataset sintético gerado via API do Google Gemini com pelo menos 50 pares de prompt/resposta, divididos em 90% treino e 10% teste.

### Passo 2 — Quantização (QLoRA)
Modelo carregado em 4-bit com `BitsAndBytesConfig` usando quantização `nf4` e `compute_dtype` float16.

### Passo 3 — LoRA
`LoraConfig` configurado com `TaskType.CAUSAL_LM`, rank `r=64`, `lora_alpha=16` e `lora_dropout=0.1`.

### Passo 4 — Treinamento
Treinamento orquestrado com `Trainer` do `transformers`, otimizador `paged_adamw_32bit`, scheduler `cosine` e `warmup_ratio=0.03`.

## Requisitos

- Python 3.10+
- GPU NVIDIA com suporte a CUDA (recomendado) ou CPU

## Nota de Integridade

Partes geradas/complementadas com IA, revisadas por Yan.