# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 將工作目錄 (CWD) 設定為目前腳本所在的目錄
Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ---------------------------------
# 1. Set shared parameters
# ---------------------------------
$EpochNum = 300
$LossValues = @(0.1, 1, 10) # PowerShell Array

# ---------------------------------
# 2. Outer loop: (unlabeled / labeled)
# ---------------------------------
foreach ($Mode in @("labeled", "unlabeled")) {

    # Set the python flag based on the $Mode
    $Flag = ""
    if ($Mode -eq "unlabeled") {
        $Flag = "--unlabeled"
    }
    
    Write-Host "" # Newline
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "===== Starting Group: $Mode (Epoch=$EpochNum, Flag=$Flag) =====" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan

    # ---------------------------------
    # 3. Mid loop: Loss Weights (Alpha)
    # ---------------------------------
    foreach ($Alpha in $LossValues) {
        
        # ---------------------------------
        # 4. Inner loop: Loss Weights (Beta)
        # ---------------------------------
        foreach ($Beta in $LossValues) {
        
            Write-Host "" # Newline
            Write-Host "--- Params: Loss=$Alpha $Beta, Epoch=$EpochNum, Flag=$Flag ---" -ForegroundColor Yellow
            
            # (Note) PowerShell's line continuation character is ` (backtick)
            
            # Time Reversal 1 (Train)
            Write-Host "[$Mode]: Training Time Reversal 1: ($Alpha, $Beta)..."
            python .\WD\WD_reverse.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2020-02-19\wireless_training.csv `
                                      --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2019-06-11\wireless_training.csv `
                                      --work_dir time_reversal_1 `
                                      --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
            
            # Time Reversal 1 (Test)
            Write-Host "[$Mode]: Testing Time Reversal 1: ($Alpha, $Beta)..."
            python .\WD\WD_reverse.py --test --work_dir time_reversal_1 `
                                      --loss_weights $Alpha $Beta --epoch $EpochNum $Flag

            # Time Reversal 2 (Train)
            Write-Host "[$Mode]: Training Time Reversal 2: ($Alpha, $Beta)..."        
            python .\WD\WD_reverse.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2020-02-19\wireless_training.csv `
                                      --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2019-10-09\wireless_training.csv `
                                      --work_dir time_reversal_2 `
                                      --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
            
            # Time Reversal 2 (Test)
            Write-Host "[$Mode]: Testing Time Reversal 2: ($Alpha, $Beta)..."
            python .\WD\WD_reverse.py --test --work_dir time_reversal_2 `
                                      --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
            
            Write-Host "-------------------------------------------------" -ForegroundColor Yellow
        }
    }
}

Write-Host "" # Newline
Write-Host "[INFO] End... All experiments finished." -ForegroundColor Green

# Use a pure English prompt
# Read-Host "Press Enter to exit..."