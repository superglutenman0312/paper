# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

# ================= Configuration =================

# 1. Define Method Names
# The script expects the file to be located at: "MethodName/MethodName.py"
# Note: Assuming 'DANN_CORR_GEMINI' follows the pattern 'DANN_CORR_GEMINI/DANN_CORR_GEMINI.py'
$methods = @(
    "DANN", 
    "DANN_CORR", 
    "DANN_CORR_GEMINI",
    "DNN"
)

# $methods = @(
#     "DANN_CORR"
# )

# 2. Define Random Seeds
$seeds = @(42, 70, 100)
# $seeds = @(42)

# 3. Define Modes
$modes = @("Labeled", "Unlabeled")
# $modes = @("Labeled")
# 4. Data Paths
$baseDataPath = "D:\paper_thesis\Histloc_real\Experiment\data"

# Define Scenarios (Time Variation vs Spatial Variation)
# $scenarios = @(
#     @{
#         Name = "Time_Variation";
#         SourceData = "$baseDataPath\220318\GalaxyA51\wireless_training.csv";
#         TargetData = "$baseDataPath\231116\GalaxyA51\wireless_training.csv";
#         DirName = "time_variation";
#         case = "1"
#     },
#     @{
#         Name = "Spatial_Variation";
#         SourceData = "$baseDataPath\231116\GalaxyA51\wireless_training.csv";
#         TargetData = "$baseDataPath\231117\GalaxyA51\wireless_training.csv";
#         DirName = "spatial_variation";
#         case = "2"
#     }
# )

$scenarios = @(
    @{
        Name = "Time_Variation";
        SourceData = "$baseDataPath\220318\GalaxyA51\wireless_training.csv";
        TargetData = "$baseDataPath\231116\GalaxyA51\wireless_training.csv";
        DirName = "time_variation";
        case = "1"
    }
)

# ================= Execution Logic =================

foreach ($method in $methods) {
    
    # Construct script path: e.g., DANN/DANN.py
    $scriptName = "${method}.py"
    $scriptPath = "$method/$scriptName"

    # Check if script exists
    if (-not (Test-Path $scriptPath)) {
        Write-Host "Warning: Cannot find script at $scriptPath ... SKIPPING." -ForegroundColor Red
        continue
    }

    foreach ($scen in $scenarios) {
        
        # Work directory argument (Relative path)
        # Result: DANN/time_variation/random_seed_xx
        $currentWorkDir = $scen.DirName

        Write-Host "########################################################" -ForegroundColor Magenta
        Write-Host "Method: $method | Scenario: $($scen.Name)" -ForegroundColor Magenta
        Write-Host "Script: $scriptPath"
        Write-Host "WorkDir: $currentWorkDir"
        Write-Host "########################################################"

        foreach ($mode in $modes) {
            
            # Determine argument flag
            if ($mode -eq "Unlabeled") {
                $modeFlag = "--unlabeled"
            } else {
                $modeFlag = ""
            }

            foreach ($seed in $seeds) {
                
                Write-Host "--------------------------------------------------------" -ForegroundColor Cyan
                Write-Host "Executing..." -ForegroundColor Yellow
                Write-Host "Mode: $mode | Seed: $seed" -ForegroundColor Yellow
                
                # --- 1. Training ---
                python $scriptPath --training_source_domain_data $($scen.SourceData) `
                                   --training_target_domain_data $($scen.TargetData) `
                                   --work_dir $currentWorkDir `
                                   --random_seed $seed --case $($scen.case) --epoch 1000 `
                                   $modeFlag

                # --- 2. Testing ---
                python $scriptPath --test `
                                   --work_dir $currentWorkDir `
                                   --random_seed $seed --case $($scen.case) --epoch 1000 `
                                   $modeFlag

                Write-Host "Done." -ForegroundColor Gray
            }
        }
    }
}
# Write-Host "[INFO] Ploting MDE diagram..." -ForegroundColor Green
# python plot_others_mde.py

Write-Host "########################################################" -ForegroundColor Magenta
Write-Host "ALL TASKS COMPLETED." -ForegroundColor Magenta
Write-Host "########################################################"