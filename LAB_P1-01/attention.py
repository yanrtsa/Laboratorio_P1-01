import numpy as np


def __init__(self):
    pass

def softmax(x):
    x_stable = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_stable)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(q,k,v) -> np.ndarray:

    d_k = k.shape[1] # Pegando o numero de colunas de K
    scores = np.matmul(q, k.T) # Multiplicacao de Q vezes a transposta de K
    scaled_scores = scores / np.sqrt(d_k) # Pegando a multiplicação e dividindo pela raiz do numero de colunas de K
    attention_weights = softmax(scaled_scores) # Normaliza os valores de 0 a 1
    output = np.matmul(attention_weights, v)

    return output