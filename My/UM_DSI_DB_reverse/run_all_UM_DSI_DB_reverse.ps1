# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

Write-Host "[Parent] Calling Child Script..."

# Write-Host "[1/3] Running MALL experiments..."
# & "$PSScriptRoot\run_WD_UM_DSI_DB_reverse.ps1"

Write-Host "[2/3] Running MALL experiments..."
& "$PSScriptRoot\run_SWD_UM_DSI_DB_reverse.ps1"

Write-Host "[3/3] Running MALL experiments..."
& "$PSScriptRoot\run_others_UM_DSI_DB_reverse.ps1"

Write-Host "Ploting MDE diagram..."
python "$PSScriptRoot\plot_all_mde2.py"

Write-Host "ALL DONE"