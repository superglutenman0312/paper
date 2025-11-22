# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 將工作目錄 (CWD) 設定為目前腳本所在的目錄
Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ---------------------------------
# 1. Set parameters 
# ---------------------------------
$EpochNum = 500                                     # (保留原本的 Epoch 設定)
$Alpha = 1                                          # Alpha 固定為 1
$BetaValues = @(0.001, 0.01, 0.1, 1, 10, 100, 1000) # Beta 範圍擴大
$Seeds = @(42, 70, 100)                             # Random Seed 列表
# $EpochNum = 1
# $Alpha = 1                                      # Alpha 固定為 1
# $BetaValues = @(0.1, 1) # Beta 範圍擴大
# $Seeds = @(42, 70)                         # 新增 Random Seed 列表

# ---------------------------------
# 2. Outer loop: (unlabeled / labeled)
# ---------------------------------
foreach ($Mode in @("labeled", "unlabeled")) {
# foreach ($Mode in @("unlabeled")) {

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
            
            # ==========================================
            # Experiment 1: Time Variation
            # ==========================================
            
            # (Train)
            Write-Host "[$Mode]: Training Time Variation ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --training_source_domain_data D:/paper_thesis/Histloc_real/Experiment/data/220318/GalaxyA51/wireless_training.csv `
                                  --training_target_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_training.csv `
                                  --work_dir time_variation2 `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag
            
            # (Test)
            Write-Host "[$Mode]: Testing Time Variation ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --test --work_dir time_variation2 `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag

            # ==========================================
            # Experiment 2: Spatial Variation
            # ==========================================

            # (Train)
            Write-Host "[$Mode]: Training Spatial Variation ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --training_source_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_training.csv `
                                  --training_target_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231117/GalaxyA51/wireless_training.csv `
                                  --work_dir spatial_variation2 `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum --random_seed $Seed $Flag
            
            # (Test)
            Write-Host "[$Mode]: Testing Spatial Variation ($Alpha, $Beta, Seed=$Seed)..."
            python .\SWD\SWD.py --test --work_dir spatial_variation2 `
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
