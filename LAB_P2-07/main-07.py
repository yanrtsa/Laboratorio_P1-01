import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-qlora-adapter"
TRAIN_FILE = "train.json"
TEST_FILE  = "test.json"


def load_dataset_file(path: str, tokenizer, max_length=512) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            records = json.loads(content)
        else:
            for line in content.splitlines():
                if line.strip():
                    records.append(json.loads(line))

    texts = []
    for item in records:
        prompt   = item.get("prompt",   item.get("instruction", ""))
        response = item.get("response", item.get("output",      ""))
        texts.append(f"### Instrução:\n{prompt}\n\n### Resposta:\n{response}")

    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    encodings["labels"] = encodings["input_ids"].copy()
    return Dataset.from_dict(encodings)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_dataset = load_dataset_file(TRAIN_FILE, tokenizer)
test_dataset  = load_dataset_file(TEST_FILE,  tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)