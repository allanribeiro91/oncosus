# Encerra tudo que estiver escutando na porta 8000 (uvicorn / API antiga).
for ($i = 0; $i -lt 6; $i++) {
    $conns = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
    if (-not $conns) { break }
    $ids = $conns | ForEach-Object { $_.OwningProcess } | Select-Object -Unique
    foreach ($procId in $ids) {
        if ($procId -gt 0) {
            Write-Host "Encerrando PID $procId..."
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 400
}
Start-Sleep -Seconds 1
$left = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($left) {
    Write-Warning "Ainda ha processo na porta 8000. Feche o terminal onde rodou uvicorn (Ctrl+C) ou use o Gerenciador de Tarefas."
} else {
    Write-Host "Porta 8000 livre."
}
