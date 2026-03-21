"""
train_model.py
Fine-tuning do BioMistral-7B com QLoRA para o OncoSUS.
Lê configurações de finetuning/training_config.yaml
"""
import gc, json, os, torch, yaml
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Autenticação HuggingFace via variável de ambiente
token = os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)

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
    train = train.select(range(min(600, len(train))))
    val   = val.select(range(min(60, len(val))))
    print(f"✅ Dataset: {len(train)} treino | {len(val)} validação")
    return train, val

def load_model():
    print(f"🤖 Carregando {MODEL_ID} em 4-bit NF4...")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM disponível: {vram:.1f} GB")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=L["r"], lora_alpha=L["alpha"],
        target_modules=L["target_modules"],
        lora_dropout=L["dropout"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
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
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing_kwargs"):
            m.gradient_checkpointing_kwargs = {"use_reentrant": False}

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
        logging_steps=5,
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

    print(f"\n🚀 Iniciando fine-tuning OncoSUS...")
    print(f"   Modelo        : {MODEL_ID}")
    print(f"   Steps         : {T['max_steps']}")
    print(f"   Batch efetivo : {T['per_device_train_batch_size'] * T['gradient_accumulation_steps']}")
    print(f"   Max length    : {T['max_length']} tokens")
    print(f"   Início        : {datetime.now().strftime('%H:%M:%S')}")
    print()

    trainer.train()
    print(f"\n✅ Fine-tuning concluído: {datetime.now().strftime('%H:%M:%S')}")
    return trainer

def salvar_adapter(trainer, tokenizer):
    adapter_path = "finetuning/output/final_adapter"
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Adaptadores LoRA salvos em: {adapter_path}")

    # Mostra os arquivos gerados
    for f in Path(adapter_path).iterdir():
        size_kb = f.stat().st_size / 1e3
        print(f"   {f.name:<40} {size_kb:.0f} KB")

if __name__ == "__main__":
    # Libera memória antes de começar
    gc.collect()
    torch.cuda.empty_cache()

    vram_livre = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated()
    ) / 1e9
    print(f"VRAM livre: {vram_livre:.1f} GB")

    if vram_livre < 6:
        print("❌ VRAM insuficiente. Feche outros programas e tente novamente.")
        exit(1)

    train_ds, val_ds = load_datasets()
    model, tokenizer = load_model()
    trainer          = train(model, tokenizer, train_ds, val_ds)
    salvar_adapter(trainer, tokenizer)