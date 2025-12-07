# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray
Write-Host "Run Child Script"

$scriptTimer = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "[1/4] Running UM_DSI_DB2 experiments..."
& "$PSScriptRoot\UM_DSI_DB2\run_all_UM_DSI_DB2.ps1"

Write-Host "[2/4] Running UM_DSI_DB_reverse2 experiments..."
& "$PSScriptRoot\UM_DSI_DB_reverse2\run_all_UM_DSI_DB_reverse2.ps1"

Write-Host "[3/4] Running MALL2 experiments..."
& "$PSScriptRoot\MALL2\run_all_MALL2.ps1"

Write-Host "[4/4] Running MCSL2_2 experiments..."
& "$PSScriptRoot\MCSL2_2\run_all_MCSL2_2.ps1"

$timeSpan = $scriptTimer.Elapsed
Write-Host "--------------------------------------------------------" -ForegroundColor Green
Write-Host "Total Execution Time: $($timeSpan.ToString("hh\:mm\:ss"))" -ForegroundColor Green
Write-Host "--------------------------------------------------------"

Write-Host "ALL Done"