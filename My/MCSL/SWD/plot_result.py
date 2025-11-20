import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 從 evaluator.py 複製的核心函式 ---
# (我們將 evaluator.py 的功能內建到這個腳本中)

def class_to_coordinate(a):
    """
    使用 NumPy 2D 陣列 (table) 將標籤 ID 轉換為 (x, y) 座標。
    """
    table = np.array([
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0,11, 0, 0, 0, 0,21],
        [ 0, 2, 0, 0, 0, 0,12, 0, 0, 0, 0,22],
        [ 0, 3, 0, 0, 0, 0,13, 0, 0, 0, 0,23],
        [ 0, 4, 0, 0, 0, 0,14, 0, 0, 0, 0,24],
        [ 0, 5, 0, 0, 0, 0,15, 0, 0, 0, 0,25],
        [ 0, 6, 0, 0, 0, 0,16, 0, 0, 0, 0,26],
        [ 0, 7, 0, 0, 0, 0,17, 0, 0, 0, 0,27],
        [ 0, 8, 0, 0, 0, 0,18, 0, 0, 0, 0,28],
        [ 0, 9, 0, 0, 0, 0,19, 0, 0, 0, 0,29],
        [ 0,10, 0, 0, 0, 0,20, 0, 0, 0, 0,30],
        [ 0,31,32,33,34,35,36,37,38,39,40,41]
    ], dtype=int)
    
    locations = np.argwhere(table == a)
    if locations.size == 0:
        raise KeyError(f"標籤 {a} 在 table 中找不到。")
        
    x = locations[0][1]  # 欄索引 (column index)
    y = locations[0][0]  # 列索引 (row index)
    coordinate = [x, y]
    return coordinate

def euclidean_distance(p1, p2):
    """
    計算歐幾里得距離，並乘以 0.6 的縮放因子。
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) * 0.6

def calculate_mde_from_file(file_path):
    """
    讀取指定的預測 CSV 檔案，並計算 MDE。
    """
    if not os.path.exists(file_path):
        print(f"  [警告] 找不到檔案: {file_path}。跳過。")
        return np.nan # 返回 NaN

    try:
        results = pd.read_csv(file_path)
        if 'label' not in results.columns or 'pred' not in results.columns:
            print(f"  [錯誤] CSV 檔案 {file_path} 缺少 'label' 或 'pred' 欄位。")
            return np.nan
    except Exception as e:
        print(f"  [錯誤] 讀取 CSV 檔案 {file_path} 時發生錯誤: {e}")
        return np.nan

    errors = []
    for idx, row in results.iterrows():
        try:
            pred_label = int(row['pred'])
            actual_label = int(row['label'])
            
            pred_coord = class_to_coordinate(pred_label)
            actual_coord = class_to_coordinate(actual_label)
            
            distance_error = euclidean_distance(pred_coord, actual_coord)
            errors.append(distance_error)
        
        except KeyError as e:
            # 標籤在 table 中找不到
            print(f"  [警告] {e}。跳過 {file_path} 中的第 {idx} 行。")
        except Exception as e:
            print(f"  [錯誤] 處理 {file_path} 第 {idx} 行時發生錯誤: {e}。")

    if errors:
        return np.mean(errors)
    else:
        return np.nan

# --- 2. 繪圖函式 ---

def plot_mde_chart(df, mode, experiment_name, output_filename):
    """
    根據指定的 mode (labeled/unlabeled) 繪製分組長條圖。
    """
    # 1. 篩選出特定 mode 的資料
    mode_df = df[df['mode'] == mode].copy()
    if mode_df.empty:
        print(f"沒有找到 {mode} 模式的資料可供繪圖。")
        return

    # 2. 準備繪圖資料
    # 取得唯一的 alpha/beta 組合標籤 (並保持順序)
    combo_labels = mode_df['combo_label'].unique()
    
    # 分別取得 Source 和 Target 的 MDE 值
    source_mdes = mode_df[mode_df['type'] == 'Source']['mde'].values
    target_mdes = mode_df[mode_df['type'] == 'Target']['mde'].values

    x = np.arange(len(combo_labels))  # 9 個 X 軸位置
    width = 0.35  # Bar 的寬度

    # 3. 開始繪圖
    # 增加圖表寬度 (18) 以容納 9 組 Bar
    fig, ax = plt.subplots(figsize=(18, 7)) 
    
    rects1 = ax.bar(x - width/2, source_mdes, width, label='Source MDE')
    rects2 = ax.bar(x + width/2, target_mdes, width, label='Target MDE')

    # 4. 設定圖表標籤
    ax.set_ylabel('Mean Distance Error (MDE)')
    ax.set_title(f'MDE Analysis for {mode.capitalize()} Models ({experiment_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=0) # 將 X 軸標籤設為 "a=0.1\nb=0.1"
    ax.legend()

    # 5. 在 Bar 上顯示 MDE 數值
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"\n成功儲存圖表: {output_filename}")


# --- 3. 主程式：循環計算與繪圖 ---

def main():
    
    # --- (可修改) 參數設定 ---
    # 決定要分析哪個實驗 ("time_variation" 或 "spatial_variation")
    EXPERIMENT_TYPE = ['time_variation', 'spatial_variation'][0]
    
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [0.1, 1.0, 10.0]
    BETAS = [0.1, 1.0, 10.0]
    EPOCH = 500 # 根據您的 ps1 腳本
    
    # 根據實驗類型，設定要抓取的 "Source" 和 "Target" 檔案
    if EXPERIMENT_TYPE == 'time_variation':
        SOURCE_FILE_TAG = '220318'
        TARGET_FILE_TAG = '231116'
        print(f"開始分析 [Time Variation] 實驗 (Source: {SOURCE_FILE_TAG}, Target: {TARGET_FILE_TAG})")
    elif EXPERIMENT_TYPE == 'spatial_variation':
        SOURCE_FILE_TAG = '231116'
        TARGET_FILE_TAG = '231117'
        print(f"開始分析 [Spatial Variation] 實驗 (Source: {SOURCE_FILE_TAG}, Target: {TARGET_FILE_TAG})")
    else:
        print(f"錯誤：未知的 EXPERIMENT_TYPE: {EXPERIMENT_TYPE}")
        return
        
    base_work_dir = EXPERIMENT_TYPE
    all_results = []

    # --- 循環計算 MDE ---
    for mode in MODES:
        for alpha in ALPHAS:
            for beta in BETAS:
                
                # 組合出資料夾路徑，例如: "time_variation/0.1_1_5_labeled"
                params_str = f"{alpha}_{beta}_{EPOCH}_{mode}"
                model_dir = os.path.join(base_work_dir, params_str, 'predictions')
                
                # 組合 X 軸標籤 (用換行\n讓圖表更美觀)
                combo_label = f"a={alpha} b={beta}"

                print(f"\n正在處理: {model_dir}")

                # 1. 計算 Source MDE
                source_path = os.path.join(model_dir, f'{SOURCE_FILE_TAG}_results.csv')
                mde_source = calculate_mde_from_file(source_path)
                
                # 2. 計算 Target MDE
                target_path = os.path.join(model_dir, f'{TARGET_FILE_TAG}_results.csv')
                mde_target = calculate_mde_from_file(target_path)

                # 3. 儲存結果
                all_results.append({
                    'mode': mode,
                    'alpha': alpha,
                    'beta': beta,
                    'combo_label': combo_label,
                    'type': 'Source',
                    'mde': mde_source
                })
                all_results.append({
                    'mode': mode,
                    'alpha': alpha,
                    'beta': beta,
                    'combo_label': combo_label,
                    'type': 'Target',
                    'mde': mde_target
                })

    # --- 轉換為 DataFrame 並繪圖 ---
    if not all_results:
        print("錯誤：沒有收集到任何結果。")
        return
        
    results_df = pd.DataFrame(all_results)
    
    # 顯示 MDE 表格 (可選)
    print("\n--- 完整 MDE 結果 (NaN 代表檔案缺失) ---")
    print(results_df)

    # 繪製兩張圖
    plot_mde_chart(results_df, 'labeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_labeled_mde_results.png')
    plot_mde_chart(results_df, 'unlabeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_unlabeled_mde_results.png')


if __name__ == "__main__":
    main()