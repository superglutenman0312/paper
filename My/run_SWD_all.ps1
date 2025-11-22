# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$startTime = Get-Date

Write-Host "[Parent] Calling Child Script..."

Write-Host "[1/4] Running MALL experiments..."
& "$PSScriptRoot\MALL\run_SWD_MALL.ps1"

Write-Host "[2/4] Running UM_DSI_DB experiments..."
& "$PSScriptRoot\UM_DSI_DB\run_SWD_UM_DSI_DB.ps1"

Write-Host "[3/4] Running UM_DSI_DB_reverse experiments..."
& "$PSScriptRoot\UM_DSI_DB_reverse\run_SWD_UM_DSI_DB_reverse.ps1"

Write-Host "[4/4] Running MCSL experiments..."
& "$PSScriptRoot\MCSL\run_SWD_MCSL.ps1"

Write-Host "[Parent] ALL Child Script done."

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "It takes $($duration.Hours)hours $($duration.Minutes)minutes $($duration.Seconds)seconds to finish all experiments."