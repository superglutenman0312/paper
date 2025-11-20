# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "[Parent] Calling Child Script..."

# Write-Host "[1/4] Running MALL experiments..."
# & "$PSScriptRoot\MALL\run_script_MALL.ps1"

Write-Host "[2/4] Running UM_DSI_DB experiments..."
& "$PSScriptRoot\UM_DSI_DB\run_script_UM_DSI_DB.ps1"

Write-Host "[3/4] Running UM_DSI_DB_reverse experiments..."
& "$PSScriptRoot\UM_DSI_DB_reverse\run_script_UM_DSI_DB_reverse.ps1"

Write-Host "[4/4] Running MCSL experiments..."
& "$PSScriptRoot\MCSL\run_script_MCSL.ps1"

Write-Host "[Parent] Child Script done."