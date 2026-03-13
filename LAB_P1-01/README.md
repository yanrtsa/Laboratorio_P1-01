# Laboratório P1-01

# Scaled Dot-Product Attention

Implementação do mecanismo descrito no paper
"Attention Is All You Need".

## Como executar

1. Instale as dependências:
   pip install -r requirements.txt

2. Execute o teste:
   python test_attention.py

## Explicação da Normalização

A matriz QK^T é dividida por √d_k para evitar que os valores
cresçam excessivamente com o aumento da dimensão das chaves.
Isso mantém a estabilidade numérica do Softmax.

## Exemplo

Input:

Q = [[1,0,1],[0,1,0]]
K = [[1,0,1],[0,1,0]]
V = [[1,2],[3,4]]

Output esperado:

[[1.88, 2.88],
 [2.62, 3.62]]

Foi usado ferramenta de ia como o chat gpt para tirar duvidas.