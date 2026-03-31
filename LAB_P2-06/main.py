# Laboratório 6 - BPE Tokenizer

# Corpus inicial
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
    print("Vocab inicial:")
    for word, freq in vocab.items():
        print(f"{word}: {freq}")

    stats = get_stats(vocab)

    best_pair = max(stats, key=stats.get)

    print("\nPar mais frequente:", best_pair)

    new_vocab = merge_vocab(best_pair, vocab)

    print("\nNovo vocab após merge:")
    for word, freq in new_vocab.items():
        print(f"{word}: {freq}")


if __name__ == "__main__":
    main()