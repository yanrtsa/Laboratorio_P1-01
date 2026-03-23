import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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


class SimpleTranslationDataset(Dataset):
    def __init__(self, input_ids, decoder_input_ids, decoder_target_ids):
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        self.decoder_target_ids = decoder_target_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.decoder_input_ids[idx], dtype=torch.long),
            torch.tensor(self.decoder_target_ids[idx], dtype=torch.long),
        )


def tokenize_dataset(pairs, max_length=50):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    input_ids = []
    decoder_input_ids = []
    decoder_target_ids = []

    for src, tgt in pairs:
        src_tokens = tokenizer.encode(
            src,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        tgt_tokens = tokenizer.encode(
            tgt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 2,
        )

        tgt_in = [tokenizer.cls_token_id] + tgt_tokens
        tgt_out = tgt_tokens + [tokenizer.sep_token_id]

        tgt_in = tgt_in[:max_length]
        tgt_out = tgt_out[:max_length]

        # Padding com pad_token_id
        src_padded = src_tokens + [tokenizer.pad_token_id] * (max_length - len(src_tokens))
        tgt_in_padded = tgt_in + [tokenizer.pad_token_id] * (max_length - len(tgt_in))
        tgt_out_padded = tgt_out + [tokenizer.pad_token_id] * (max_length - len(tgt_out))

        input_ids.append(src_padded)
        decoder_input_ids.append(tgt_in_padded)
        decoder_target_ids.append(tgt_out_padded)

    return tokenizer, input_ids, decoder_input_ids, decoder_target_ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, max_len=50, pad_id=0):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

    def forward(self, src, tgt):
        # src: [batch, seq] -> [seq, batch]
        # tgt: [batch, seq] -> [seq, batch]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_embed = self.positional_encoding(src_embed)
        tgt_embed = self.positional_encoding(tgt_embed)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        src_padding_mask = (src == self.pad_id).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_id).transpose(0, 1)

        memory = self.transformer.encoder(src_embed, src_key_padding_mask=src_padding_mask)

        out = self.transformer.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        logits = self.fc_out(out)
        return logits.transpose(0, 1)  # [batch, seq, vocab]


def train_loop(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for src, tgt_in, tgt_out in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            optimizer.zero_grad()

            output = model(src, tgt_in)
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_out_flat = tgt_out.reshape(-1)

            loss = criterion(output_flat, tgt_out_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")


def greedy_decode(model, tokenizer, src_ids, max_length=50, device="cpu"):
    model.eval()

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq]
    tgt = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long, device=device)  # [1, 1]

    with torch.no_grad():
        for _ in range(max_length - 1):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == tokenizer.sep_token_id:
                break

    return tgt.squeeze().tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tarefa 1: dataset real (subset)
    pairs = load_small_dataset(limit=1000)
    tokenizer, input_ids, decoder_input_ids, decoder_target_ids = tokenize_dataset(pairs, max_length=50)

    dataset = SimpleTranslationDataset(input_ids, decoder_input_ids, decoder_target_ids)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Tarefa 3: modelo + loss + optimizer
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        max_len=50,
        pad_id=pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Iniciando treinamento...\n")
    train_loop(model, dataloader, criterion, optimizer, device, epochs=10)

    # Tarefa 4: overfitting test (reconhecer frase do treino)
    sample_src, sample_tgt = pairs[0]
    print(f"\nFrase de treinamento (fonte): {sample_src}")
    print(f"Frase de treinamento (alvo): {sample_tgt}")

    sample_src_ids = tokenizer.encode(sample_src, add_special_tokens=False, truncation=True, max_length=50)
    sample_src_ids = sample_src_ids + [pad_id] * (50 - len(sample_src_ids))

    generated_ids = greedy_decode(model, tokenizer, sample_src_ids, max_length=50, device=device)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nTradução gerada (auto-regressiva): {generated_text}")
    print(f"Tokens gerados: {generated_tokens}")

    print("\nTreinamento finalizado. Verifique se loss caiu ao longo de epochs e se a tradução memorizou a frase de treino.")