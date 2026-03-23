# LAB_P2-05: Treinamento Fim-a-Fim do Transformer

## Objetivo
Este notebook implementa o Lab 05 solicitado: usar dataset real do Hugging Face + tokenização + transformer + loop de treinamento PyTorch + teste de overfitting no conjunto de treino.

## Arquivos principais
- `LAB_P2-05/main.py`: pipeline completo de 1) carregamento de dados, 2) tokenização, 3) construção de modelo, 4) treinamento, 5) inferência autoregressiva.
- `LAB_P1-04/main.py`: implementação original do Transformer do Lab 04 (não alterada por este lab). 

## Dependências
Ativar virtualenv e instalar pacotes:

```powershell
cd c:\Users\fryhh\PycharmProjects\Laboratorio_P1-01
.venv\Scripts\Activate.ps1
pip install datasets transformers torch torchvision torchaudio
```

> Se o ambiente for CPU-only, usar:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

## Tarefa 1: dataset real
- Dataset utilizado: `bentrevett/multi30k` (split `train`)
- Subset seleccionado: primeiros 1000 exemplos
- Mapeamento: inglês (`en`) -> alemão (`de`)

## Tarefa 2: tokenização
- Tokenizador Hugging Face: `bert-base-multilingual-cased`
- Entrada: `input_ids` com padding/truncation tamanho fixo (`max_length=50`)
- Target decoder:
  - `decoder_input_ids`: `[CLS] + tokens`
  - `decoder_target_ids`: `tokens + [SEP]`
  - `pad_token_id` em ambos para comprimento fixo

## Tarefa 3: motor de otimização
- Modelo: `SimpleTransformer` (PyTorch)
  - d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2
- Loss: `CrossEntropyLoss(ignore_index=pad_token_id)`
- Otimizador: `Adam(lr=1e-3)`
- Loop: 10 epochs (pode ajustar em `train_loop`)

## Tarefa 4: prova de fogo / overfitting
- Seleciona um exemplo do treinamento (`pairs[0]`)
- Executa `greedy_decode` autoregressivo para esse frase
- Exibe saída gerada e comparação com frase original
- Resultado: demonstra que o modelo aprendeu ao menos a memorização local

## Executar

```powershell
python LAB_P2-05/main.py
```

### O que observar
- Output de cada epoch com `Loss: ...` decrescendo
- Tradução final de frase de treino com tokens e texto gerado

## Observações
1. Este trabalho foca no fluxo de dados, iterador treinável e convergência no lab de CPU.
2. O pipeline em `LAB_P2-05/main.py` é independente de `LAB_P1-04/main.py` e preserva a implementação original do Lab 4.

## Ferramentas de IA Utilizadas
Conforme instruções do laboratório, utilizamos IA para facilitar a manipulação dos datasets e a tokenização (Tarefas 1 e 2):
- **Chat GPT**: Assistência na implementação da integração com Hugging Face datasets e configuração do AutoTokenizer

O fluxo de Forward/Backward da Tarefa 3 interage estritamente com as classes construídas nos laboratórios anteriores (LAB_P1-01 a LAB_P1-04), sem utilização de IA para essa parte específica.
