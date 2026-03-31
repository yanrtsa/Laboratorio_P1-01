# Laboratório 6 - Tokenização com BPE e WordPiece

## 📌 Descrição

Este projeto implementa o algoritmo Byte Pair Encoding (BPE) do zero, simulando o processo de construção de um vocabulário baseado em sub-palavras. Além disso, é realizada uma integração com o tokenizador WordPiece do BERT utilizando a biblioteca Hugging Face Transformers.

---

## ⚙️ Parte 1 e 2: Implementação do BPE

O algoritmo BPE foi implementado em duas etapas principais:

- **Contagem de pares (`get_stats`)**: identifica a frequência de pares adjacentes de símbolos no vocabulário.
- **Fusão de pares (`merge_vocab`)**: combina os pares mais frequentes em novos tokens.

O processo é repetido por 5 iterações, permitindo observar a formação de sub-palavras como:


est</w>


Esse comportamento demonstra como o modelo aprende padrões morfológicos comuns.

---

## 🤖 Parte 3: WordPiece com BERT

Foi utilizado o tokenizador:


bert-base-multilingual-cased


Para processar a frase:


Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar.


### 🔍 Exemplo de saída:


['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er', 'são',
'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f', '##í', '##cei', '##s',
'de', 'aj', '##usta', '##r', '.']


---

## 🧠 O que significa `##`?

No WordPiece, o prefixo `##` indica que o token é uma **continuação da palavra anterior**.

### Exemplo:


in + ##cons + ##titucional + ##mente


Forma a palavra:


inconstitucionalmente


---

## 💡 Por que isso é importante?

O uso de sub-palavras permite que o modelo:

- 📉 Reduza o tamanho do vocabulário
- 🧩 Compreenda palavras desconhecidas (out-of-vocabulary)
- 🔤 Generalize melhor padrões linguísticos

Mesmo que uma palavra nunca tenha sido vista durante o treinamento, o modelo pode interpretá-la a partir de partes conhecidas.

---

## ⚠️ Uso de IA

Ferramentas de IA generativa foram utilizadas como apoio na implementação, especialmente na construção das funções de manipulação de strings. Todo o código foi revisado e compreendido antes da entrega, conforme exigido pelas instruções do laboratório.

---

## 🚀 Execução

```bash
pip install transformers
python main.py