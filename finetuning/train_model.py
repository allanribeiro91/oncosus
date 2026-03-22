"""
train_model.py — OncoSUS
Fine-tuning do Llama 3.2 3B Instruct com QLoRA para oncologia do SUS.
"""
import gc, json, os, torch, yaml
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Autenticação HuggingFace
token = os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("✅ Autenticado no HuggingFace")

with open("finetuning/training_config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MODEL_ID   = CFG["model"]["base_model"]
T, L       = CFG["training"], CFG["lora"]
TRAIN_PATH = CFG["data"]["train_path"]
VAL_PATH   = CFG["data"]["val_path"]


def load_datasets():
    def load(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    train = Dataset.from_list(load(TRAIN_PATH))
    val   = Dataset.from_list(load(VAL_PATH))

    # Llama 3.2 3B é menor — pode usar mais exemplos por step
    train = train.select(range(min(3000, len(train))))
    val   = val.select(range(min(150,  len(val))))

    print(f"✅ Dataset: {len(train)} treino | {len(val)} validação")
    return train, val


def load_model():
    print(f"🤖 Carregando {MODEL_ID} em 4-bit NF4...")

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM disponível: {vram:.1f} GB")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=L["r"],
        lora_alpha=L["alpha"],
        target_modules=L["target_modules"],
        lora_dropout=L["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)

    # Llama 3.2 já tem pad_token — não precisa adicionar
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"   Parâmetros treináveis: {treinaveis:,} ({100*treinaveis/total:.3f}%)")
    return model, tokenizer


def train(model, tokenizer, train_ds, val_ds):
    out_dir = Path(T["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = False
    model.enable_input_require_grads()

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        max_steps=T["max_steps"],
        per_device_train_batch_size=T["per_device_train_batch_size"],
        gradient_accumulation_steps=T["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=T["learning_rate"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=T["eval_steps"],
        save_strategy="steps",
        save_steps=T["save_steps"],
        load_best_model_at_end=True,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        warmup_steps=T["warmup_steps"],
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=T["max_length"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        formatting_func=lambda ex: ex["text"],
    )

    print(f"\n🚀 Iniciando fine-tuning OncoSUS — Llama 3.2 3B Instruct")
    print(f"   Steps         : {T['max_steps']}")
    print(f"   Batch efetivo : {T['per_device_train_batch_size'] * T['gradient_accumulation_steps']}")
    print(f"   Max length    : {T['max_length']} tokens")
    print(f"   Início        : {datetime.now().strftime('%H:%M:%S')}")
    print()

    trainer.train()
    print(f"\n✅ Concluído: {datetime.now().strftime('%H:%M:%S')}")
    return trainer


def salvar_adapter(trainer, tokenizer):
    adapter_path = "finetuning/output/final_adapter"
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Adapter salvo em: {adapter_path}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    vram_livre = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated()
    ) / 1e9
    print(f"VRAM livre: {vram_livre:.1f} GB")

    if vram_livre < 5:
        print("❌ VRAM insuficiente. Feche outros programas.")
        exit(1)

    train_ds, val_ds = load_datasets()
    model, tokenizer = load_model()
    trainer          = train(model, tokenizer, train_ds, val_ds)
    salvar_adapter(trainer, tokenizer)