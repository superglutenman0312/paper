# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

# ================= Configuration =================

# 1. Define Method Names
# The script expects the file to be located at: "MethodName/MethodName_reverse.py"
$methods = @(
    "DANN", 
    "DANN_CORR"
)

# 2. Define Random Seeds
$seeds = @(42, 70, 100)

# 3. Define Modes
$modes = @("Labeled", "Unlabeled")

# 4. Data Paths
# Note: Using 'tony_data_reverse' as requested
$baseDataPath = "D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse"

# Source Domain is fixed for Time Reversal: 2020-02-19
$sourceData = "$baseDataPath\2020-02-19\wireless_training.csv"

# Define Scenarios
$scenarios = @(
    @{
        Name = "Time_Reversal_1";
        TargetData = "$baseDataPath\2019-06-11\wireless_training.csv";
        DirName = "time_reversal_1"
    },
    @{
        Name = "Time_Reversal_2";
        TargetData = "$baseDataPath\2019-10-09\wireless_training.csv";
        DirName = "time_reversal_2"
    }
)

# ================= Execution Logic =================

foreach ($method in $methods) {
    
    # Construct script path: e.g., DANN/DANN_reverse.py
    $scriptName = "${method}_reverse.py"
    $scriptPath = "$method/$scriptName"

    # Check if script exists
    if (-not (Test-Path $scriptPath)) {
        Write-Host "Warning: Cannot find script at $scriptPath ... SKIPPING." -ForegroundColor Red
        continue
    }

    foreach ($scen in $scenarios) {
        
        # Work directory argument (Relative path)
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
                python $scriptPath --training_source_domain_data $sourceData `
                                   --training_target_domain_data $($scen.TargetData) `
                                   --work_dir $currentWorkDir `
                                   --random_seed $seed `
                                   $modeFlag

                # --- 2. Testing ---
                python $scriptPath --test `
                                   --work_dir $currentWorkDir `
                                   --random_seed $seed `
                                   $modeFlag

                Write-Host "Done." -ForegroundColor Gray
            }
        }
    }
}
Write-Host "[INFO] Ploting MDE diagram..." -ForegroundColor Green
python plot_others_mde.py

Write-Host "########################################################" -ForegroundColor Magenta
Write-Host "ALL TASKS COMPLETED." -ForegroundColor Magenta
Write-Host "########################################################"