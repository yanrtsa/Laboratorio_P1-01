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


def main():
    print("Vocab inicial:")
    for word, freq in vocab.items():
        print(f"{word}: {freq}")

    print("\nPares e frequências:")
    stats = get_stats(vocab)

    for pair, freq in stats.items():
        print(f"{pair}: {freq}")

    # validação pedida no PDF
    print("\nFrequência de ('e', 's'):", stats.get(('e', 's')))


if __name__ == "__main__":
    main()