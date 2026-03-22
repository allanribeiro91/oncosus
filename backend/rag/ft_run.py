from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/ft_dataset/ft_dataset.jsonl"
)

def format_chat(example):

    system = example["messages"][0]["content"]
    user = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]

    text = f"""<s>[INST] {system}

{user} [/INST]
{assistant}</s>"""

    return {"text": text}

dataset = dataset.map(format_chat)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-oncosus",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()


trainer.model.save_pretrained("lora-oncosus")
tokenizer.save_pretrained("lora-oncosus")

