from datasets import load_dataset


def load_small_dataset(dataset_name="bentrevett/multi30k", split="train", limit=1000):
    dataset = load_dataset(dataset_name, split=split)

    subset = dataset.select(range(limit))

    pairs = []
    for item in subset:
        src = item["en"]
        tgt = item["de"]
        pairs.append((src, tgt))

    return pairs


# teste
if __name__ == "__main__":
    data = load_small_dataset(limit=10)
    for i in range(3):
        print(data[i])