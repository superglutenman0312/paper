# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray
Write-Host "Run Child Script"

Write-Host "[1/2] Running SWD experiments..."
& "$PSScriptRoot\run_SWD_MALL2.ps1"

Write-Host "[2/2] Running others experiments..."
& "$PSScriptRoot\run_others_MALL2.ps1"

Write-Host "Ploting MDE diagram..."
python "$PSScriptRoot\plot_all_mde_cdf.py"

Write-Host "ALL Done"