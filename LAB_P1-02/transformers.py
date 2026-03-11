import numpy as np
import pandas as pd

vocab = {"o":0, "banco":1, "bloqueou":2, "cartao":3}

df_vocab = pd.DataFrame(list(vocab.items()), columns=["palavra","id"])
print("Vocabulário:")
print(df_vocab)

# Frase de entrada
sentence = ["o","banco","bloqueou","o","cartao"]

ids = [vocab[word] for word in sentence]
print("ids:")
print(ids)

vocab_size = len(vocab)
d_model = 64
batch_size = 1
seq_len = len(ids)

embeddings = np.random.randn(vocab_size, d_model)

X = embeddings[ids]

X = X.reshape(batch_size, seq_len, d_model)

print("\nFormato da entrada:", X.shape)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def self_attention(X):

    batch, tokens, d_model = X.shape
    dk = d_model

    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = Q @ K.transpose(0,2,1)

    # Scaling
    scores = scores / np.sqrt(dk)

    attn = softmax(scores)

    # SANITY CHECK
    soma_att = attn.sum(axis=-1)
    print("Sanity check - soma da atenção (deve ser ~1):")
    print(soma_att)

    Z = attn @ V

    return Z

def feed_forward(X):

    batch, tokens, d_model = X.shape
    d_ff = 256

    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)

    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    hidden = X @ W1 + b1
    hidden = np.maximum(0, hidden)

    out = hidden @ W2 + b2

    return out

def encoder_layer(X):

    X_att = self_attention(X)

    X_norm1 = layer_norm(X + X_att)

    X_ffn = feed_forward(X_norm1)

    X_out = layer_norm(X_norm1 + X_ffn)

    return X_out

N = 6

for i in range(N):
    X = encoder_layer(X)
    print(f"Saída da camada {i+1}:", X.shape)


Z = X

print("\nRepresentação final Z:")
print(Z)
print("Shape final:", Z.shape)