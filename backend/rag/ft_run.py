from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FT_DATASET = PROJECT_ROOT / "backend/data/ft_dataset/ft_dataset.jsonl"


# ----------------------------------
# 1. LOAD DATASET
# ----------------------------------
dataset = load_dataset(
    "json",
    data_files=str(FT_DATASET)
)

# ----------------------------------
# 2. FORMAT CHAT (Mistral style)
# ----------------------------------
def format_chat(example):

    system = example["messages"][0]["content"]
    user = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]

    # 🔥 formato correto para Mistral Instruct
    text = f"""<s>[INST] {system}

{user} [/INST] {assistant}</s>"""

    return {"text": text}


dataset = dataset.map(format_chat)

# ----------------------------------
# 3. TOKENIZER + MODEL
# ----------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# ----------------------------------
# 4. LoRA CONFIG
# ----------------------------------
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ----------------------------------
# 5. TRAINING CONFIG
# ----------------------------------
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-oncosus",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",  # 🔥 melhor para 4bit
    report_to="none"
)

# ----------------------------------
# 6. TRAINER
# ----------------------------------
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    args=training_args,
    max_seq_length=2048  # 🔥 importante para seu contexto longo
)

# ----------------------------------
# 7. TRAIN
# ----------------------------------
trainer.train()

# ----------------------------------
# 8. SAVE
# ----------------------------------
trainer.model.save_pretrained("lora-oncosus")
tokenizer.save_pretrained("lora-oncosus")