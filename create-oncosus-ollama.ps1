# Cria ou atualiza o modelo local `oncosus-llm` a partir do Modelfile.
# Se `ollama run` ficar no spinner: pare a API (Ctrl+C) para liberar RAM, rode este script, depois teste de novo.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".\Modelfile")) {
    Write-Error "Modelfile nao encontrado em $PSScriptRoot"
    exit 1
}
$env:OLLAMA_NUM_GPU = "0"
Write-Host ">>> ollama create oncosus-llm -f Modelfile"
ollama create oncosus-llm -f .\Modelfile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host ">>> Modelos instalados:"
ollama list
Write-Host ""
Write-Host ">>> Teste minimo (em CPU pode levar 30s-3min na 1ª vez). Se travar, feche a API e outros apps pesados."
ollama run oncosus-llm "Responda exatamente uma palavra: ok."
