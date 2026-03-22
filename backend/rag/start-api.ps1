# API com limites de RAM conservadores para Ollama (CPU).
# Menor contexto = menos RAM no Ollama junto com embeddings na mesma máquina
$env:ONCOSUS_OLLAMA_NUM_CTX = "512"
$env:ONCOSUS_OLLAMA_NUM_BATCH = "128"
if (-not $env:ONCOSUS_TOP_K) { $env:ONCOSUS_TOP_K = "4" }
if (-not $env:ONCOSUS_FINAL_K) { $env:ONCOSUS_FINAL_K = "2" }
Set-Location $PSScriptRoot
Write-Host "Subindo API em http://127.0.0.1:8000"
Write-Host "Dica: se der erro 1455 ao carregar embeddings, feche o Ollama da bandeja neste momento, espere a API subir, depois abra o Ollama."
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
