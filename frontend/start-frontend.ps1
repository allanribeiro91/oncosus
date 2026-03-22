# Sobe o Angular sempre nesta pasta (não use "cd ...\frontend" — "..." não é caminho válido no PowerShell).
Set-Location $PSScriptRoot
if (-not (Test-Path ".\package.json")) {
    Write-Error "Execute este script de dentro de oncosus-novo\frontend (package.json sumiu)."
    exit 1
}
if (-not (Test-Path ".\node_modules")) {
    Write-Host "Rodando npm install..."
    npm install
}
# Libera 4200 para nao cair em porta aleatoria (52619, 57748...) — URL antiga deixa de responder.
& "$PSScriptRoot\stop-front.ps1"
Write-Host "Front: http://127.0.0.1:4200 (proxy /api -> 127.0.0.1:8000)"
npx --yes ng serve --port 4200 --host 127.0.0.1
