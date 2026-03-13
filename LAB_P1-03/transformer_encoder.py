import numpy as np

def create_causal_mask(seq_len):
    mask = np.full((seq_len, seq_len), 0, dtype=float)
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Tarefa 1: Teste da Máscara Causal
seq_len = 5
mask = create_causal_mask(seq_len)
print("Máscara causal:")
print(mask)

# matrizes fictícias
Q = np.random.rand(seq_len, 4)
K = np.random.rand(seq_len, 4)

scores = Q @ K.T

masked_scores = scores + mask

attention = softmax(masked_scores)

print("Scores com máscara:")
print(attention)

# Tarefa 2: Cross-Attention
def cross_attention(encoder_out, decoder_state):
    d_model = encoder_out.shape[-1]

    W_q = np.random.rand(d_model, d_model)
    W_k = np.random.rand(d_model, d_model)
    W_v = np.random.rand(d_model, d_model)
    
    Q = decoder_state[0] @ W_q
    K = encoder_out[0] @ W_k
    V = encoder_out[0] @ W_v
    
    d_k = d_model
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    attention_weights = softmax(scores)
    output = attention_weights @ V 
    return output[None, :, :] 

# Tensores fictícios
batch_size = 1
seq_len_frances = 10
d_model = 512
encoder_output = np.random.rand(batch_size, seq_len_frances, d_model)

seq_len_ingles = 4
decoder_state = np.random.rand(batch_size, seq_len_ingles, d_model)

output = cross_attention(encoder_output, decoder_state)
print("Output shape da Cross-Attention:", output.shape)

# Tarefa 3: Simulação do Loop Auto-Regressivo
vocab_size = 10000
vocab = [f"token_{i}" for i in range(vocab_size)]
vocab[0] = "<START>"
vocab[1] = "O"
vocab[2] = "rato"
vocab[9999] = "<EOS>"

def generate_next_token(current_sequence, encoder_out):
    # Mock: retorna probabilidades aleatórias
    probs = np.random.rand(vocab_size)
    probs = probs / np.sum(probs)
    return probs

# Loop de inferência
current_sequence = ["<START>", "O", "rato"]
encoder_out = encoder_output  # usando o tensor fictício

while True:
    probs = generate_next_token(current_sequence, encoder_out)
    next_token_idx = np.argmax(probs)
    next_token = vocab[next_token_idx]
    current_sequence.append(next_token)
    if next_token == "<EOS>":
        break
    if len(current_sequence) > 20:  # para evitar loop infinito no mock
        break

print("Frase final gerada:", " ".join(current_sequence))