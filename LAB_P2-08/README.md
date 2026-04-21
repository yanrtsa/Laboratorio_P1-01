# Laboratório 08 — Alinhamento Humano com DPO

Pipeline de alinhamento do modelo **TinyLlama-1.1B-Chat** utilizando Direct Preference Optimization (DPO) para garantir comportamento HHH (Helpful, Honest, Harmless).

## Estrutura do Projeto

```
LAB_P2-08/
├── main-08.py           # Pipeline DPO completo
├── hhh_dataset.jsonl    # 30 pares de preferência (15 segurança + 15 tom corporativo)
└── README.md
```

## Como Executar (Google Colab)

```python
# 1. Clone o repositório
!git clone <url-do-repo>
%cd Laboratorio_P1-01

# 2. Instale as dependências
!pip install -r requirements.txt

# 3. Execute o pipeline
!python LAB_P2-08/main-08.py
```

## Implementação

### Dataset de Preferências

30 pares no formato `prompt / chosen / rejected`:
- **15 pares de segurança**: prompts maliciosos com respostas de recusa (`chosen`) vs. conteúdo prejudicial (`rejected`).
- **15 pares de tom corporativo**: prompts profissionais com resposta adequada (`chosen`) vs. resposta rude ou inadequada (`rejected`).

### Pipeline DPO

1. Carregamento do dataset `.jsonl` como `datasets.Dataset`
2. TinyLlama carregado em 4-bit (NF4) via `BitsAndBytesConfig`
3. Adaptador LoRA aplicado via `get_peft_model()` → modelo **ator** (pesos atualizados)
4. `ref_model=None`: `DPOTrainer` utiliza internamente o modelo base com adaptador desativado como **modelo de referência congelado**, calculando a divergência KL sem necessidade de uma segunda instância em memória
5. Treinamento com `DPOTrainer` e `DPOConfig(beta=0.1)`
6. Validação: prompts maliciosos passados pelo modelo alinhado para comprovar a supressão

## O Papel Matemático do Parâmetro β (Beta)

O parâmetro β controla o equilíbrio entre o aprendizado de preferências e a fidelidade ao modelo de linguagem original. A função de perda do DPO é:

**L_DPO = −E\[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)))\]**

onde `y_w` é a resposta *chosen* e `y_l` é a resposta *rejected*. O termo `log π_θ / π_ref` mede o quanto a política aprendida se afastou da distribuição de referência. O β atua como um **"imposto sobre o desvio"**: valores altos de β penalizam fortemente qualquer afastamento do modelo de referência, preservando a fluência e coerência linguística do modelo original mas limitando o alinhamento; valores baixos permitem que a otimização de preferências seja mais agressiva, correndo o risco de degradar a qualidade geral da linguagem. Com `β = 0.1`, o modelo tem liberdade suficiente para aprender a suprimir respostas prejudiciais a partir de um dataset pequeno (30 exemplos), sem sofrer *catastrophic forgetting* das capacidades linguísticas adquiridas no pré-treinamento.

## Requisitos

- Python 3.10+
- GPU NVIDIA (Google Colab recomendado)

## Nota de Integridade

Partes geradas/complementadas com IA, revisadas por Yan.
