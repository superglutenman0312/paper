# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray

Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ================= 設定區 =================
# 1. 定義要執行的 Python 腳本列表
$scripts = @(
    "DANN/DANN_pytorch.py", 
    "DANN_CORR/DANN_CORR_MALL.py", 
    "DANN_CORR_GEMINI/DANN_CORR_MALL_gemini.py"
)

# 2. 定義 Random Seeds (那三個種子碼)
$seeds = @(42, 70, 100)

# 3. 定義模式 (Labeled vs Unlabeled)
# 邏輯：Labeled 不加參數，Unlabeled 加上 --unlabeled
$modes = @("Labeled", "Unlabeled")

# 4. 資料路徑設定 (避免底下指令太長)
$sourceData = "D:/paper_thesis/My/data/MTLocData/Mall/2021-11-20/wireless_training.csv"
$targetData = "D:/paper_thesis/My/data/MTLocData/Mall/2022-12-21/wireless_training.csv"
$workDir = "experiments"

# ================= 執行邏輯 =================
foreach ($script in $scripts) {
    foreach ($mode in $modes) {
        
        # 處理 Labeled/Unlabeled 的參數字串
        # 如果是 Unlabeled，參數字串為 "--unlabeled"
        # 如果是 Labeled，參數字串為 "" (空字串)
        if ($mode -eq "Unlabeled") {
            $modeFlag = "--unlabeled"
        } else {
            $modeFlag = ""
        }

        foreach ($seed in $seeds) {
            
            Write-Host "========================================================" -ForegroundColor Cyan
            Write-Host "Running: $script" -ForegroundColor Yellow
            Write-Host "mode: $mode ($modeFlag) | Seed: $seed" -ForegroundColor Yellow
            Write-Host "========================================================" -ForegroundColor Cyan

            # --- 1. 訓練 (Training) ---
            Write-Host "Step 1: Training..." -ForegroundColor Green
            
            # 組合訓練指令
            python $script --training_source_domain_data $sourceData `
                           --training_target_domain_data $targetData `
                           --work_dir $workDir `
                           --random_seed $seed `
                           $modeFlag

            # --- 2. 測試 (Testing) ---
            Write-Host "Step 2: Testing..." -ForegroundColor Green
            
            # 組合測試指令
            python $script --test `
                           --work_dir $workDir `
                           --random_seed $seed `
                           $modeFlag
            
            Write-Host "Done: $script - $mode - Seed $seed" -ForegroundColor Gray
            Write-Host ""
        }
    }
}

Write-Host "[INFO] Ploting MDE diagram..." -ForegroundColor Green
python plot_others_mde.py

Write-Host "--------------------------------------------------------" -ForegroundColor Magenta
Write-Host "ALL done."
Write-Host "--------------------------------------------------------"