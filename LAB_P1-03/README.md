# Laboratório P1-03

## Como executar o código

1. Instale as dependências necessárias:

```
pip install numpy pandas
```

2. Execute o arquivo Python:

```
python transformer_encoder.py
```

3. O programa irá:

- Criar e testar a máscara causal (Look-Ahead Mask) com tensores fictícios Q e K, provando que as probabilidades das palavras futuras se tornam 0.0.
- Implementar a atenção cross-attention entre encoder e decoder, calculando a saída com tensores fictícios.
- Simular o loop de inferência auto-regressiva, gerando uma sequência de tokens até o limite ou <EOS>.

## Observação

Ferramentas de Inteligência Artificial foram utilizadas apenas para tirar dúvidas durante o desenvolvimento do laboratório.