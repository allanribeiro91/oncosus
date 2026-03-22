# Encerra o que estiver escutando nas portas do ng serve (padrão: 4200).
param(
    [int[]]$Port = @(4200)
)

function Clear-ListenPort([int]$LocalPort) {
    for ($i = 0; $i -lt 6; $i++) {
        $conns = @(Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue)
        if (-not $conns) { break }
        $ids = $conns | ForEach-Object { $_.OwningProcess } | Select-Object -Unique
        foreach ($procId in $ids) {
            if ($procId -gt 0) {
                Write-Host "Encerrando PID $procId (porta $LocalPort)..."
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Milliseconds 400
    }
}

foreach ($p in $Port) {
    Clear-ListenPort $p
}
Start-Sleep -Seconds 1

$stillBusy = @()
foreach ($p in $Port) {
    if (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) {
        $stillBusy += $p
    }
}
if ($stillBusy.Count -gt 0) {
    Write-Warning "Ainda em uso: $($stillBusy -join ', '). Feche o terminal (Ctrl+C) ou o Node no Gerenciador de Tarefas."
}
else {
    Write-Host "Porta(s) livre(s): $($Port -join ', ')."
}
