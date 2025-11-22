# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 將工作目錄 (CWD) 設定為目前腳本所在的目錄
Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ---------------------------------
# 1. Set parameters 
# ---------------------------------
$EpochNum = 100
$Alpha = 1                                          # Alpha 固定為 1
$BetaValues = @(0.001, 0.01, 0.1, 1, 10, 100, 1000) # Beta 範圍擴大
$Seeds = @(42, 70, 100)                             # Random Seed 列表

# $EpochNum = 1
# $Alpha = 1                                          # Alpha 固定為 1
# $BetaValues = @(0.1, 1) # Beta 範圍擴大
# $Seeds = @(42, 70)   
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
    # 3. Mid loop: Loss Weights (Beta)
    # ---------------------------------
    foreach ($Beta in $BetaValues) {
        
        # ---------------------------------
        # 4. Inner loop: Random Seeds
        # ---------------------------------
        foreach ($Seed in $Seeds) {
        
            Write-Host "" # Newline
            Write-Host "--- Params: LossWeights=($Alpha, $Beta), Seed=$Seed, Mode=$Mode ---" -ForegroundColor Yellow
            
            # (Note) PowerShell's line continuation character is ` (backtick)
            
            # ==========================================
            # Experiment 1: Time Variation 1
            # ==========================================
            
            # (Train)
            Write-Host "[$Mode]: Training Time Variation 1 ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-06-11\wireless_training.csv `
                              --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-10-09\wireless_training.csv `
                              --work_dir time_variation_1 `
                              --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag
            
            # (Test)
            Write-Host "[$Mode]: Testing Time Variation 1 ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --test --work_dir time_variation_1 `
                              --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag

            # ==========================================
            # Experiment 2: Time Variation 2
            # ==========================================

            # (Train)
            Write-Host "[$Mode]: Training Time Variation 2 ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-06-11\wireless_training.csv `
                              --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2020-02-19\wireless_training.csv `
                              --work_dir time_variation_2 `
                              --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag
            
            # (Test)
            Write-Host "[$Mode]: Testing Time Variation 2 ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --test --work_dir time_variation_2 `
                              --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag
            
            Write-Host "-------------------------------------------------" -ForegroundColor DarkGray
        }
    }
}

Write-Host "" # Newline
Write-Host "[INFO] End... All experiments finished." -ForegroundColor Green

# ---------------------------------
# Plot results
python .\SWD\plot_result_rs.py
Write-Host "[INFO] Process Complete." -ForegroundColor Green
