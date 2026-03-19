"""
Laboratório Técnico 04: O Transformer Completo "From Scratch"
Tarefa: Integrar todos os módulos dos labs anteriores em uma arquitetura Encoder-Decoder funcional
"""

import numpy as np

def softmax(x):
    x_stable = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_stable)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)

    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def add_and_norm(x, sublayer_output):
    return layer_norm(x + sublayer_output)


def position_wise_ffn(x, d_ff=2048):
    batch_size, seq_len, d_model = x.shape
    
    # Peso 1: d_model -> d_ff
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    
    # Peso 2: d_ff -> d_model
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)
    
    # Forward
    hidden = np.matmul(x, W1) + b1
    hidden = np.maximum(0, hidden)  # ReLU
    output = np.matmul(hidden, W2) + b2
    
    return output


def create_causal_mask(seq_len):
    mask = np.full((seq_len, seq_len), 0, dtype=float)
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

# pilha encoder

def self_attention_layer(x, d_model=512):
    batch_size, seq_len, _ = x.shape

    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01

    Q = np.matmul(x, W_q)
    K = np.matmul(x, W_k)
    V = np.matmul(x, W_v)

    attention_output, _ = scaled_dot_product_attention(Q, K, V, mask=None)
    
    return attention_output


def encoder_block(x, d_model=512, d_ff=2048):
    self_attn_output = self_attention_layer(x, d_model)
    x = add_and_norm(x, self_attn_output)

    ffn_output = position_wise_ffn(x, d_ff)
    x = add_and_norm(x, ffn_output)
    
    return x


def encoder_stack(x, num_layers=6, d_model=512, d_ff=2048):
    Z = x
    print(f"\n[ENCODER] Entrada shape: {Z.shape}")
    
    for i in range(num_layers):
        Z = encoder_block(Z, d_model, d_ff)
        print(f"  Layer {i+1} output: {Z.shape}")
    
    print(f"[ENCODER] Saída final (Z): {Z.shape}")
    return Z

def masked_self_attention_layer(y, d_model=512):
    batch_size, seq_len, _ = y.shape

    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01

    Q = np.matmul(y, W_q)
    K = np.matmul(y, W_k)
    V = np.matmul(y, W_v)

    causal_mask = create_causal_mask(seq_len)

    attention_output, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    return attention_output


def cross_attention_layer(y, Z, d_model=512):
    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01

    Q = np.matmul(y, W_q)
    K = np.matmul(Z, W_k)
    V = np.matmul(Z, W_v)

    attention_output, _ = scaled_dot_product_attention(Q, K, V, mask=None)
    
    return attention_output


def decoder_block(y, Z, d_model=512, d_ff=2048, vocab_size=None):
    masked_attn_output = masked_self_attention_layer(y, d_model)
    y = add_and_norm(y, masked_attn_output)

    cross_attn_output = cross_attention_layer(y, Z, d_model)
    y = add_and_norm(y, cross_attn_output)

    ffn_output = position_wise_ffn(y, d_ff)
    y = add_and_norm(y, ffn_output)

    if vocab_size is not None:
        W_out = np.random.randn(d_model, vocab_size) * 0.01
        logits = np.matmul(y, W_out)
        probs = softmax(logits)
        return y, logits, probs
    
    return y, None, None


def decoder_stack(y, Z, num_layers=6, d_model=512, d_ff=2048, vocab_size=None):
    Y = y
    print(f"\n[DECODER] Entrada (Y) shape: {Y.shape}")
    print(f"[DECODER] Memória do Encoder (Z) shape: {Z.shape}")
    
    for i in range(num_layers):
        Y, _, _ = decoder_block(Y, Z, d_model, d_ff, vocab_size=None)
        print(f"  Layer {i+1} output: {Y.shape}")

    if vocab_size is not None:
        W_out = np.random.randn(d_model, vocab_size) * 0.01
        logits = np.matmul(Y, W_out)
        probs = softmax(logits)
        print(f"[DECODER] Logits shape: {logits.shape}")
        print(f"[DECODER] Probabilities shape: {probs.shape}")
        return Y, logits, probs
    
    return Y, None, None

def prepare_vocabulary():
    vocab = {
        "<START>": 0,
        "<EOS>": 1,
        "Thinking": 2,
        "Machines": 3,
        "The": 4,
        "robots": 5,
        "are": 6,
        "coming": 7,
        "Robôs": 8,
        "estão": 9,
        "chegando": 10,
    }
    id_to_word = {v: k for k, v in vocab.items()}
    return vocab, id_to_word


def encode_input_sentence(sentence_tokens, vocab, d_model=512):
    batch_size = 1
    seq_len = len(sentence_tokens)

    embeddings = np.random.randn(len(vocab), d_model) * 0.01

    token_ids = [vocab[token] for token in sentence_tokens]

    input_embedding = embeddings[token_ids]
    input_embedding = input_embedding.reshape(batch_size, seq_len, d_model)
    
    return input_embedding, embeddings


def autoregressive_decoding_loop(encoder_output, embeddings, vocab, id_to_word, 
                                 num_decoder_layers=6, d_model=512, d_ff=2048,
                                 vocab_size=None, max_length=20):
    if vocab_size is None:
        vocab_size = len(vocab)

    start_token = vocab.get("<START>", 0)
    eos_token = vocab.get("<EOS>", 1)
    
    current_sequence = [start_token]
    output_tokens = ["<START>"]
    
    print(f"\n[INFERENCE] Iniciando geração automática...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max length: {max_length}")
    print(f"  Iniciando com: <START>")
    
    step = 0
    while step < max_length:
        batch_size = 1
        seq_len = len(current_sequence)

        decoder_input = embeddings[current_sequence]
        decoder_input = decoder_input.reshape(batch_size, seq_len, d_model)

        _, logits, probs = decoder_stack(
            decoder_input, 
            encoder_output,
            num_layers=num_decoder_layers,
            d_model=d_model,
            d_ff=d_ff,
            vocab_size=vocab_size
        )

        last_token_probs = probs[0, -1, :]

        next_token_id = np.argmax(last_token_probs)
        next_token = id_to_word.get(next_token_id, f"<UNK_{next_token_id}>")
        
        print(f"  Step {step+1}: {next_token} (prob: {last_token_probs[next_token_id]:.4f})")

        output_tokens.append(next_token)
        current_sequence.append(next_token_id)

        if next_token_id == eos_token:
            print(f"  [PARADA] Token <EOS> gerado!")
            break
        
        step += 1
    
    return output_tokens


def full_transformer_inference(encoder_input, sentence_tokens, 
                              num_encoder_layers=6, num_decoder_layers=6,
                              d_model=512, d_ff=2048):
    vocab, id_to_word = prepare_vocabulary()
    vocab_size = len(vocab)
    
    print("="*70)
    print("TRANSFORMER COMPLETO - TESTE END-TO-END")
    print("="*70)

    print(f"\n{'='*70}")
    print("FASE 1: ENCODER")
    print(f"{'='*70}")
    print(f"[INPUT] Frase: {' '.join(sentence_tokens)}")
    print(f"[INPUT] Shape do tensor: {encoder_input.shape}")

    Z = encoder_stack(encoder_input, num_encoder_layers, d_model, d_ff)

    print(f"\n{'='*70}")
    print("FASE 2: DECODER (INFERÊNCIA AUTO-REGRESSIVA)")
    print(f"{'='*70}")

    _, embeddings = encode_input_sentence(sentence_tokens, vocab, d_model)

    generated_sequence = autoregressive_decoding_loop(
        Z, embeddings, vocab, id_to_word,
        num_decoder_layers, d_model, d_ff,
        vocab_size, max_length=20
    )
    
    print(f"\n{'='*70}")
    print("RESULTADO FINAL")
    print(f"{'='*70}")
    print(f"[ENTRADA]:  {' '.join(sentence_tokens)}")
    print(f"[SAÍDA]:    {' '.join(generated_sequence)}")
    print(f"{'='*70}\n")
    
    return generated_sequence

if __name__ == "__main__":
    batch_size = 1
    d_model = 512
    d_ff = 2048
    num_encoder_layers = 2
    num_decoder_layers = 2
    vocab_size = 11

    sentence = ["Thinking", "Machines"]
    seq_len = len(sentence)
    
    # Criar tensor de entrada simulado
    encoder_input = np.random.randn(batch_size, seq_len, d_model) * 0.01

    generated_output = full_transformer_inference(
        encoder_input,
        sentence,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        d_ff=d_ff
    )
    
    print("\n[OK] Teste concluido com sucesso!")
    print(f"  Tokenizacao saida: {generated_output}")
