from datasets import load_dataset
from transformers import AutoTokenizer


def load_small_dataset(dataset_name="bentrevett/multi30k", split="train", limit=1000):
    dataset = load_dataset(dataset_name, split=split)

    subset = dataset.select(range(limit))

    pairs = []
    for item in subset:
        src = item["en"]
        tgt = item["de"]
        pairs.append((src, tgt))

    return pairs


def tokenize_dataset(pairs, max_length=50):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    input_ids = []
    target_ids = []

    for src, tgt in pairs:
        # Encoder (entrada)
        src_tokens = tokenizer.encode(
            src,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Decoder (saída)
        tgt_tokens = tokenizer.encode(
            tgt,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Adiciona START (CLS) e EOS (SEP)
        tgt_tokens = [tokenizer.cls_token_id] + tgt_tokens + [tokenizer.sep_token_id]

        # Garante tamanho fixo
        tgt_tokens = tgt_tokens[:max_length]

        input_ids.append(src_tokens)
        target_ids.append(tgt_tokens)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids
    }


# TESTE
if __name__ == "__main__":
    data = load_small_dataset(limit=10)
    tokenized = tokenize_dataset(data)

    print("Entrada (input_ids):")
    print(tokenized["input_ids"][0])

    print("\nSaída (target_ids):")
    print(tokenized["target_ids"][0])