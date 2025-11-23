# [Before Use]: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-Location $PSScriptRoot
Write-Host "[INFO] CWD has been set to: $(Get-Location)" -ForegroundColor Gray
Write-Host "[INFO] PowerShell loop experiment starting..." -ForegroundColor Green

# ================= 設定區 =================

# 1. 定義方法名稱 (資料夾名稱與 .py 檔名)
# $methods = @(
#     "DANN", 
#     "DANN_CORR", 
#     "DANN_CORR_GEMINI"
# )

$methods = @(
    "DANN_GEMINI"
)

# 2. 定義 Random Seeds
$seeds = @(42, 70, 100)

# 3. 定義模式
$modes = @("Labeled", "Unlabeled")

# 4. 資料路徑設定
$baseDataPath = "D:\paper_thesis\Histloc_real\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data"

# 固定 Source Domain (2019-06-11)
$sourceData = "$baseDataPath\2019-06-11\wireless_training.csv"

# 定義兩個 Time Variation 場景
$scenarios = @(
    @{
        Name = "Time_Variation_1";
        TargetData = "$baseDataPath\2019-10-09\wireless_training.csv";
        DirName = "time_variation_1"
    },
    @{
        Name = "Time_Variation_2";
        TargetData = "$baseDataPath\2020-02-19\wireless_training.csv";
        DirName = "time_variation_2"
    }
)

# ================= 執行邏輯 =================
foreach ($method in $methods) {
    
    # 組合 Python 腳本路徑: e.g., DANN/DANN.py
    $scriptPath = "$method/$method.py"

    # 檢查腳本是否存在
    if (-not (Test-Path $scriptPath)) {
        Write-Host "Warning: Cannot find script at $scriptPath ... SKIPPING." -ForegroundColor Red
        continue
    }

    foreach ($scen in $scenarios) {
        
        # [修正點]：移除 $method 前綴。
        # 因為您的 Python 程式似乎是相對於該程式所在位置建立資料夾。
        # 傳入 "time_variation_1" -> 程式會在 "DANN/time_variation_1" 建立
        $currentWorkDir = "$($scen.DirName)"

        Write-Host "########################################################" -ForegroundColor Magenta
        Write-Host "Method: $method | Scenario: $($scen.Name)" -ForegroundColor Magenta
        Write-Host "Script: $scriptPath"
        Write-Host "WorkDir Arg: $currentWorkDir"
        Write-Host "########################################################"

        foreach ($mode in $modes) {
            
            if ($mode -eq "Unlabeled") {
                $modeFlag = "--unlabeled"
            } else {
                $modeFlag = ""
            }

            foreach ($seed in $seeds) {
                
                Write-Host "--------------------------------------------------------" -ForegroundColor Cyan
                Write-Host "Executing..." -ForegroundColor Yellow
                Write-Host "Mode: $mode | Seed: $seed" -ForegroundColor Yellow
                
                # --- 1. 訓練 (Training) ---
                python $scriptPath --training_source_domain_data $sourceData `
                                   --training_target_domain_data $($scen.TargetData) `
                                   --work_dir $currentWorkDir `
                                   --random_seed $seed `
                                   $modeFlag

                # --- 2. 測試 (Testing) ---
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
Write-Host "ALL DONE." -ForegroundColor Magenta
Write-Host "########################################################"