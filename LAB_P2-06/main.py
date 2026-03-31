# Laboratório 6 - BPE Tokenizer
from transformers import AutoTokenizer

vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

def get_stats(vocab):
    pairs = {}

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])

            if pair not in pairs:
                pairs[pair] = 0

            pairs[pair] += freq

    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word, freq in v_in.items():
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = freq

    return v_out

def main():
    current_vocab = vocab.copy()
    num_iterations = 5

    for i in range(num_iterations):
        print(f"\n--- Iteração {i + 1} ---")

        stats = get_stats(current_vocab)

        best_pair = max(stats, key=stats.get)
        print("Par mais frequente:", best_pair, "->", "".join(best_pair))

        current_vocab = merge_vocab(best_pair, current_vocab)

        print("Vocab atualizado:")
        for word, freq in current_vocab.items():
            print(f"{word}: {freq}")

    print("\n--- WordPiece (BERT) ---")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

    tokens = tokenizer.tokenize(frase)

    print("\nFrase:")
    print(frase)

    print("\nTokens:")
    print(tokens)


if __name__ == "__main__":
    main()