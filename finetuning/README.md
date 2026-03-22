# Fine-tuning OncoSUS — Qwen2.5-3B-Instruct

## Modelo treinado

Modelo base: `Qwen/Qwen2.5-3B-Instruct`
Técnica: QLoRA (4-bit NF4, LoRA r=8 α=16)
Dataset: 4.629 exemplos dos chunks reais dos PCDTs/INCA
Steps: 300 | Loss final: 0.90 | Accuracy: 80%

## Download do adapter treinado

Os pesos do adapter (>1GB) não estão versionados no Git.
Faça o download e salve em `finetuning/output/final_adapter/`:

🔗 [Link do adapter — Google Drive ou HuggingFace]

## Como reproduzir o fine-tuning do zero
```powershell
# 1. Instala dependências
pip install -r requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install peft trl accelerate bitsandbytes>=0.46.1

# 2. Gera o dataset a partir dos chunks do RAG
python finetuning/build_clean_dataset.py

# 3. Treina o modelo (~60 min, requer GPU NVIDIA com 8GB+ VRAM)
$env:HF_TOKEN = "seu_token_huggingface"
python finetuning/train_model.py
```

## Como testar após baixar o adapter
```powershell
# 1. Indexa os documentos no ChromaDB
python backend/scripts/step_4_0_embed_chunks.py

# 2. Inicia o assistente
cd backend/rag
python main.py
```

## Estrutura dos arquivos
```
finetuning/
├── build_clean_dataset.py   # Gera dataset dos chunks do Allan
├── generate_synthetic_qa.py # Gera dados sintéticos oncológicos  
├── prepare_dataset.py       # Anonimização e formatação
├── train_model.py           # Fine-tuning QLoRA
├── training_config.yaml     # Configuração do treino
└── output/
    └── final_adapter/       # ← baixar separadamente
        ├── adapter_config.json
        └── adapter_model.safetensors  (não versionado)
```

## Configuração do fine-tuning

| Parâmetro | Valor |
|---|---|
| Modelo base | Qwen/Qwen2.5-3B-Instruct |
| LoRA r | 8 |
| LoRA alpha | 16 |
| Learning rate | 1e-4 |
| Max steps | 300 |
| Batch efetivo | 16 |
| Max length | 768 tokens |
| VRAM necessária | ~7 GB |