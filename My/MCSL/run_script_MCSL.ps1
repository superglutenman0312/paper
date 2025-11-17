# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ---------------------------------
# 1. Set shared parameters
# ---------------------------------
$EpochNum = 5
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
            
            # Time Variation (Train)
            Write-Host "[$Mode]: Training Time Variation ($Alpha, $Beta)..."
            python .\WD\WD.py --training_source_domain_data D:/paper_thesis/Histloc_real/Experiment/data/220318/GalaxyA51/wireless_training.csv `
                                  --training_target_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_training.csv `
                                  --work_dir time_variation `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
            
            # Time Variation (Test)
            Write-Host "[$Mode]: Testing Time Variation ($Alpha, $Beta)..."
            python .\WD\WD.py --test --work_dir time_variation `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum $Flag

            # Spatial Variation (Train)
            Write-Host "[$Mode]: Training Spatial Variation ($Alpha, $Beta)..."
            python .\WD\WD.py --training_source_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_training.csv `
                                  --training_target_domain_data D:/paper_thesis/Histloc_real/Experiment/data/231117/GalaxyA51/wireless_training.csv `
                                  --work_dir spatial_variation `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
            
            # Spatial Variation (Test)
            Write-Host "[$Mode]: Testing Spatial Variation ($Alpha, $Beta)..."
            python .\WD\WD.py --test --work_dir spatial_variation `
                                  --loss_weights $Alpha $Beta --epoch $EpochNum $Flag
                                  
            Write-Host "-------------------------------------------------" -ForegroundColor Yellow
        }
    }
}

Write-Host "" # Newline
Write-Host "[INFO] End... All experiments finished." -ForegroundColor Green

# Use a pure English prompt
Read-Host "Press Enter to exit..."