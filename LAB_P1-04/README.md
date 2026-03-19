# Laboratório Técnico 04: O Transformer Completo "From Scratch"

## 📋 Descrição

Implementação completa de uma arquitetura **Transformer Encoder-Decoder** do zero, integrando módulos desenvolvidos nos laboratórios anteriores. O projeto demonstra o fluxo completo de um Transformer, desde o processamento da entrada no Encoder até a geração automática de tokens no Decoder através de um loop auto-regressivo.

### Objetivos de Aprendizagem

1. ✅ Integrar módulos de redes neurais separados em uma topologia coerente
2. ✅ Garantir fluxo correto de tensores através de camadas Add & Norm e Feed-Forward
3. ✅ Acoplar o loop auto-regressivo de inferência na saída do Decoder

---

## 🏗️ Arquitetura do Transformer

### Componentes Implementados

#### **Tarefa 1: Blocos de Montar**
- `scaled_dot_product_attention(Q, K, V, mask)` - Mecanismo de atenção escalada
- `layer_norm(x)` - Normalização de camada
- `add_and_norm(x, sublayer)` - Conexão residual com normalização
- `position_wise_ffn(x, d_ff)` - Rede feed-forward com ReLU
- `create_causal_mask(seq_len)` - Máscara causal para o Decoder

#### **Tarefa 2: Pilha do Encoder**
- `self_attention_layer(x)` - Auto-atenção bidirecional
- `encoder_block(x)` - Bloco completo do Encoder
- `encoder_stack(x, num_layers)` - Pilha de N blocos

Fluxo de um EncoderBlock:
```
Entrada X
    ↓
Self-Attention (Q, K, V da mesma fonte)
    ↓
Add & Norm
    ↓
Feed-Forward Network (d_model → d_ff → d_model)
    ↓
Add & Norm
    ↓
Saída Z (contexto bidirecional)
```

#### **Tarefa 3: Pilha do Decoder**
- `masked_self_attention_layer(y)` - Auto-atenção com máscara causal
- `cross_attention_layer(y, Z)` - Cross-atenção (Decoder ↔ Encoder)
- `decoder_block(y, Z)` - Bloco completo do Decoder
- `decoder_stack(y, Z, num_layers)` - Pilha de N blocos + Linear → Softmax

Fluxo de um DecoderBlock:
```
Entrada Y (sequência do Decoder)
    ↓
Masked Self-Attention (máscara causal impede visão futura)
    ↓
Add & Norm
    ↓
Cross-Attention (Q do Decoder, K,V do Encoder)
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
Linear Projection → Softmax (probabilidades sobre vocabulário)
```

#### **Tarefa 4: Inferência Auto-regressiva**
- `prepare_vocabulary()` - Vocabulário fictício
- `encode_input_sentence()` - Codificação de entrada
- `autoregressive_decoding_loop()` - Geração sequencial de tokens
- `full_transformer_inference()` - Teste end-to-end

Fluxo de Inferência:
```
"Thinking Machines" 
    ↓
[ENCODER] × 2 layers
    ↓ Saída: Z (contexto)
    ↓
[DECODER] Loop Auto-regressivo:
  Iteração 1: <START> → prediz token 1
  Iteração 2: <START> token1 → prediz token 2
  Iteração 3: ... → prediz <EOS> (parada)
    ↓
Saída final: "<START> token1 token2 <EOS>"
```

---

## 📚 Dependências

### Bibliotecas Utilizadas

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| **NumPy** | ≥ 1.20 | Operações matemáticas e manipulação de matrizes |

### Arquivo `requirements.txt`

```
numpy>=1.20.0
```

---

## 🚀 Como Instalar e Executar

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes)

### Instalação

#### 1. Clonar o repositório
```bash
git clone <seu-repositorio>
cd Laboratorio_P1-01/LAB_P1-04
```

#### 2. Criar ambiente virtual (recomendado)
```bash
python -m venv venv
```

#### 3. Ativar ambiente virtual

**No Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**No Windows (CMD):**
```cmd
venv\Scripts\activate
```

**No Linux/Mac:**
```bash
source venv/bin/activate
```

#### 4. Instalar dependências
```bash
pip install -r requirements.txt
```

### Executar o Programa

```bash
python main.py
```

### Saída Esperada

```
======================================================================
TRANSFORMER COMPLETO - TESTE END-TO-END
======================================================================

======================================================================
FASE 1: ENCODER
======================================================================
[INPUT] Frase: Thinking Machines
[INPUT] Shape do tensor: (1, 2, 512)

[ENCODER] Entrada shape: (1, 2, 512)
  Layer 1 output: (1, 2, 512)
  Layer 2 output: (1, 2, 512)
[ENCODER] Saida final (Z): (1, 2, 512)

======================================================================
FASE 2: DECODER (INFERENCIA AUTO-REGRESSIVA)
======================================================================

[INFERENCE] Iniciando geracao automatica...
  Vocab size: 11
  Max length: 20
  Iniciando com: <START>

[DECODER] Entrada (Y) shape: (1, 1, 512)
[DECODER] Memoria do Encoder (Z) shape: (1, 2, 512)
  Layer 1 output: (1, 1, 512)
  Layer 2 output: (1, 1, 512)
[DECODER] Logits shape: (1, 1, 11)
[DECODER] Probabilities shape: (1, 1, 11)
  Step 1: token (prob: 0.xxxx)
  ...
  Step N: <EOS> (prob: 0.xxxx)
  [PARADA] Token <EOS> gerado!

======================================================================
RESULTADO FINAL
======================================================================
[ENTRADA]:  Thinking Machines
[SAIDA]:    <START> token1 token2 ... <EOS>
======================================================================

[OK] Teste concluido com sucesso!
  Tokenizacao saida: ['<START>', 'token1', 'token2', ..., '<EOS>']
```

---

## 📊 Estrutura do Codigo

```
main.py
├─ TAREFA 1: Blocos de Montar
│  ├─ softmax(x)
│  ├─ scaled_dot_product_attention(Q, K, V, mask)
│  ├─ layer_norm(x, eps)
│  ├─ add_and_norm(x, sublayer_output)
│  ├─ position_wise_ffn(x, d_ff)
│  └─ create_causal_mask(seq_len)
│
├─ TAREFA 2: Pilha do Encoder
│  ├─ self_attention_layer(x, d_model)
│  ├─ encoder_block(x, d_model, d_ff)
│  └─ encoder_stack(x, num_layers, d_model, d_ff)
│
├─ TAREFA 3: Pilha do Decoder
│  ├─ masked_self_attention_layer(y, d_model)
│  ├─ cross_attention_layer(y, Z, d_model)
│  ├─ decoder_block(y, Z, d_model, d_ff, vocab_size)
│  └─ decoder_stack(y, Z, num_layers, d_model, d_ff, vocab_size)
│
├─ TAREFA 4: Inferencia Auto-regressiva
│  ├─ prepare_vocabulary()
│  ├─ encode_input_sentence(sentence_tokens, vocab, d_model)
│  ├─ autoregressive_decoding_loop(...)
│  └─ full_transformer_inference(encoder_input, sentence_tokens, ...)
│
└─ MAIN: Teste end-to-end
   └─ if __name__ == "__main__"
```

---

## 🔧 Parametrização

Os seguintes parâmetros podem ser ajustados no final do arquivo `main.py`:

```python
batch_size = 1           # Tamanho do lote
d_model = 512            # Dimensão do modelo (embedding)
d_ff = 2048              # Dimensão intermediária da FFN
num_encoder_layers = 2   # Número de blocos no Encoder
num_decoder_layers = 2   # Número de blocos no Decoder
vocab_size = 11          # Tamanho do vocabulário
max_length = 20          # Comprimento máximo de geração
```

---

## 🧮 Conceitos Matemáticos Implementados

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Layer Normalization
$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### Add & Norm (Conexão Residual)
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### Position-wise Feed-Forward
$$\text{FFN}(x) = \text{Linear}(\text{ReLU}(\text{Linear}(x)))$$

---

## 📖 Estrutura do Projeto Completo

Este laboratório integra módulos de:
- **LAB_P1-01**: Scaled Dot-Product Attention
- **LAB_P1-02**: Causal Masking e Cross-Attention
- **LAB_P1-03**: Transformer Encoder com Add & Norm e FFN
- **LAB_P1-04**: Arquitetura Encoder-Decoder completa + Inferência

---

## ✅ Validação do Código

O programa executa as seguintes validações:

1. **Dimensionalidade dos tensores**: Verifica se Input, Encoder Output e Decoder Output têm as dimensões corretas
2. **Sanity checks**: Confirma que pesos de atenção somam 1 (após softmax)
3. **Loop auto-regressivo**: Valida que a geração para com token `<EOS>`
4. **Integração end-to-end**: Testa o fluxo completo Encoder → Decoder → Geração

---

## 📝 Notas sobre Uso de Inteligência Artificial

**Partes geradas/complementadas com IA, revisadas por [Seu Nome]**

A IA foi utilizada apenas dentro do escopo permitido:
- ✅ **Brainstorming** de estrutura modular
- ✅ **Template básico** das funções
- ✅ **Documentação** e comentários explicativos

**NÃO foram gerados integralmente pela IA:**
- ❌ A lógica matemática da Scaled Dot-Product Attention (baseada em LAB_P1-01)
- ❌ A implementação da máscara causal (baseada em LAB_P1-02)
- ❌ A estrutura do Encoder com Add & Norm (baseada em LAB_P1-03)
- ❌ O loop auto-regressivo do Decoder
- ❌ A integração entre Encoder e Decoder

Toda a lógica foi **compreendida, estudada e validada**, garantindo que o código implementado reflete o entendimento completo dos conceitos de Transformers.

---

## 🔗 Referências

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- NumPy Documentation: https://numpy.org/doc/
- Laboratórios P1-01, P1-02, P1-03

---

## 📞 Suporte

Para dúvidas sobre a implementação, consulte o código comentado em `main.py` ou revise os laboratórios anteriores.

---

**Última atualização**: Março 18, 2026
