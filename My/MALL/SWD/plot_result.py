import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle  # (新增) 匯入 pickle 模組

# --- 1. (新增) 載入並反轉座標/標籤對照表 (來自您的範例) ---
# (請根據您的 'label_map.pkl' 檔案實際位置修改此路徑)
LABEL_MAP_PATH = 'D:/paper_thesis/My/data/MTLocData/Mall/label_map.pkl' 

try:
    # 載入 { 座標(tuple): 標籤(int) }
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f) 
    # 反轉字典，變為 { 標籤(int): 座標(tuple) }
    LABEL_TO_COORDINATE_DICT = {value: key for key, value in label_map.items()} 
    print(f"成功載入並反轉座標對照表: {LABEL_MAP_PATH}")
except FileNotFoundError:
    print(f"錯誤：找不到 'label_map.pkl' 檔案於: {LABEL_MAP_PATH}")
    print("請檢查此分析腳本中的 LABEL_MAP_PATH 變數是否設定正確。")
    exit()
except Exception as e:
    print(f"載入 {LABEL_MAP_PATH} 時發生錯誤: {e}")
    exit()


# --- 2. (修改) 'Evaluator' class 中的「標籤/座標 對照表」 ---
def class_to_coordinate(a):
    """
    (已修改) 使用從 .pkl 載入的字典 (dict) 將標籤 ID 轉換為 (x, y) 座標。
    """
    try:
        # LABEL_TO_COORDINATE_DICT 是一個從 .pkl 載入的全域變數
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        # 如果 CSV 中的 label 不在 .pkl 字典中
        raise KeyError(f"標籤 {a} 在 {LABEL_MAP_PATH} 載入的字典中找不到。")

# --- 3. (修改) 'Evaluator' class 中的「距離計算公式」 ---
def euclidean_distance(p1, p2):
    """
    (已修改) 計算標準歐幾里得距離 (使用 np.linalg.norm)。
    p1, p2 均為 (x, y) 格式的座標。
    """
    # 使用您 .pkl 範例程式碼中的 np.linalg.norm
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- 4. (保留) 計算 MDE 的主函式 ---
# (此函式無需修改，它會自動呼叫上面更新過的函式)
def calculate_mde_from_file(file_path):
    """
    讀取指定的預測 CSV 檔案，並計算 MDE。
    """
    if not os.path.exists(file_path):
        print(f"  [警告] 找不到檔案: {file_path}。跳過。")
        return np.nan 

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
            # (注意) 根據您的 .pkl 範例，標籤可能是 int 或 float
            # 我們統一轉為 int (因為 pkl 的 key 是 int)
            pred_label = int(row['pred'])
            actual_label = int(row['label'])
            
            # (自動) 呼叫新的 'pkl dict' 查表法
            pred_coord = class_to_coordinate(pred_label)
            actual_coord = class_to_coordinate(actual_label)
            
            # (自動) 呼叫新的 'np.linalg.norm' 距離公式
            distance_error = euclidean_distance(pred_coord, actual_coord)
            errors.append(distance_error)
        
        except KeyError as e:
            # 標籤在 pkl 字典中找不到
            print(f"  [警告] {e}。跳過 {file_path} 中的第 {idx} 行。")
        except Exception as e:
            print(f"  [錯誤] 處理 {file_path} 第 {idx} 行時發生錯誤: {e}。")

    if errors:
        return np.mean(errors)
    else:
        return np.nan

# --- 5. (保留) 繪圖函式 ---
# (此函式無需修改)
def plot_mde_chart(df, mode, experiment_name, output_filename):
    """
    根據指定的 mode (labeled/unlabeled) 繪製分組長條圖。
    """
    mode_df = df[df['mode'] == mode].copy()
    if mode_df.empty:
        print(f"沒有找到 {mode} 模式的資料可供繪圖。")
        return

    combo_labels = mode_df['combo_label'].unique()
    source_mdes = mode_df[mode_df['type'] == 'Source']['mde'].values
    target_mdes = mode_df[mode_df['type'] == 'Target']['mde'].values

    x = np.arange(len(combo_labels))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(18, 7)) 
    rects1 = ax.bar(x - width/2, source_mdes, width, label='Source MDE')
    rects2 = ax.bar(x + width/2, target_mdes, width, label='Target MDE')

    # (修改) 更新標題，標註 MDE 來源
    ax.set_ylabel('Mean Distance Error (MDE)')
    ax.set_title(f'MDE Analysis for {mode.capitalize()} Models ({experiment_name}) - (Using Pickle Map)')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=0)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"\n成功儲存圖表: {output_filename}")


# --- 6. (保留) 主程式：循環計算與繪圖 ---
# (此函式無需修改，它會自動抓取您實驗資料夾中的檔案)
def main():
    
    # --- (可修改) 參數設定 ---
    # 確保這些設定與您新資料集的實驗資料夾結構一致
    EXPERIMENT_TYPE = 'experiments' 
    
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [0.1, 1.0, 10.0]
    BETAS = [0.1, 1.0, 10.0]
    EPOCH = 100
    
    # !! 重要 !!
    # 這裡的 'SOURCE_FILE_TAG' 和 'TARGET_FILE_TAG' 
    # 必須對應到您 *新資料集* 實驗的 predictions 資料夾中的 .csv 檔名
    # 例如，如果新資料集的檔名是 'source_data_results.csv' 和 'target_data_results.csv'
    # 您需要修改 '220318' -> 'source_data'
    
    if EXPERIMENT_TYPE == 'experiments':
        SOURCE_FILE_TAG = '211120' # (範例)
        TARGET_FILE_TAG = '221221' # (範例)
        print(f"開始分析 [experiments] 實驗 (Source: {SOURCE_FILE_TAG}, Target: {TARGET_FILE_TAG})")
    else:
        print(f"錯誤：未知的 EXPERIMENT_TYPE: {EXPERIMENT_TYPE}")
        return
        
    base_work_dir = EXPERIMENT_TYPE
    all_results = []

    for mode in MODES:
        for alpha in ALPHAS:
            for beta in BETAS:
                
                params_str = f"{alpha}_{beta}_{EPOCH}_{mode}"
                model_dir = os.path.join(base_work_dir, params_str, 'predictions')
                combo_label = f"a={alpha}\nb={beta}"

                print(f"\n正在處理: {model_dir}")

                # 1. 計算 Source MDE
                source_path = os.path.join(model_dir, f'{SOURCE_FILE_TAG}_results.csv')
                mde_source = calculate_mde_from_file(source_path)
                
                # 2. 計算 Target MDE
                target_path = os.path.join(model_dir, f'{TARGET_FILE_TAG}_results.csv')
                mde_target = calculate_mde_from_file(target_path)

                # 3. 儲存結果
                all_results.append({ 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Source', 'mde': mde_source })
                all_results.append({ 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Target', 'mde': mde_target })

    if not all_results:
        print("錯誤：沒有收集到任何結果。")
        return
        
    results_df = pd.DataFrame(all_results)
    
    print("\n--- 完整 MDE 結果 (NaN 代表檔案缺失) ---")
    print(results_df)

    # (修改) 儲存為新檔名
    plot_mde_chart(results_df, 'labeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_labeled_mde_results_Pickle.png')
    plot_mde_chart(results_df, 'unlabeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_unlabeled_mde_results_Pickle.png')


if __name__ == "__main__":
    main()