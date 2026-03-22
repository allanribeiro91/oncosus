# Mesmo que start-frontend.ps1, mas porta 4201 (quando 4200 ficar para outro app).
Set-Location $PSScriptRoot
if (-not (Test-Path ".\package.json")) {
    Write-Error "Execute este script de dentro de oncosus-novo\frontend (package.json sumiu)."
    exit 1
}
if (-not (Test-Path ".\node_modules")) {
    Write-Host "Rodando npm install..."
    npm install
}
& "$PSScriptRoot\stop-front.ps1" -Port 4201
Write-Host "Front: http://127.0.0.1:4201 (proxy /api -> 127.0.0.1:8000)"
npx --yes ng serve --port 4201 --host 127.0.0.1
