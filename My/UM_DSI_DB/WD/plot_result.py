import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. (修改) 新資料集的「標籤/座標 對照表」 (字典) ---
LABEL_TO_COORDINATE_DICT = {
    1: (-53.56836, 5.83747), 2: (-50.051947, 5.855995), 3: (-46.452556, 5.869534),
    4: (-42.853167, 5.883073), 5: (-44.659589, 7.011051), 6: (-44.751032, 11.879306),
    7: (-40.626278, 11.865147), 8: (-37.313205, 14.650224), 9: (-40.672748, 7.050528),
    10: (-39.253777, 5.896612), 11: (-35.654387, 5.91015), 12: (-32.054999, 5.923687),
    13: (-29.658016, 7.136601), 14: (-29.715037, 10.176074), 15: (-28.455609, 5.937224),
    16: (-24.856221, 5.950761), 17: (-21.256833, 5.964297), 18: (-21.06986, 12.146254),
    19: (-17.657445, 5.977833), 20: (-14.058057, 5.991368), 21: (-14.001059, 12.211117),
    22: (-10.458671, 6.004903), 23: (-6.859283, 6.018437), 24: (-6.616741, 8.258015),
    25: (-3.259896, 6.031971), 26: (0.33949, 6.045505), 27: (0.297446, 12.26954),
    28: (3.938876, 6.059038), 29: (7.538262, 6.07257), 30: (7.525253, 12.321256),
    31: (11.137647, 6.086102), 32: (14.737032, 6.099633), 33: (14.705246, 2.374095),
    34: (14.717918, 12.321068), 35: (18.336417, 6.113164), 36: (21.935801, 6.126695),
    37: (21.899795, 12.339099), 38: (21.921602, 2.423358), 39: (36.238672, 6.108184),
    40: (32.733952, 6.167284), 41: (31.779903, 2.442016), 42: (29.134569, 6.153754),
    43: (25.535185, 6.140225), 44: (38.088066, 7.394376), 45: (38.040971, 10.951591),
    46: (37.993873, 14.508804), 47: (29.037591, 12.318132), 48: (44.93136, 6.314889),
    49: (44.816113, 13.54513)
}

def class_to_coordinate(a):
    """
    (已修改) 使用字典 (dict) 將標籤 ID 轉換為 (x, y) 座標。
    """
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        raise KeyError(f"標籤 {a} 在 LABEL_TO_COORDINATE_DICT 中找不到。")

# --- 2. (修改) 新資料集的「距離計算公式」 (標準歐幾里得) ---
def euclidean_distance(p1, p2):
    """
    (已修改) 計算標準歐幾里得距離 (Standard Euclidean distance)。
    (已移除 0.6 縮放因子)
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- 3. (保留) 計算 MDE 的主函式 ---
# (此函式無需修改，它會自動呼叫上面更新過的函式)
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
            
            # (自動) 呼叫新的 'dict' 查表法
            pred_coord = class_to_coordinate(pred_label)
            actual_coord = class_to_coordinate(actual_label)
            
            # (自動) 呼叫新的 '標準' 距離公式
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

# --- 4. (保留) 繪圖函式 ---
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

    # (修改) 更新標題，移除 0.6 縮放的標註
    ax.set_ylabel('Mean Distance Error (MDE)')
    ax.set_title(f'MDE Analysis for {mode.capitalize()} Models ({experiment_name}) - Standard Euclidean')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=0)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"\n成功儲存圖表: {output_filename}")


# --- 5. (保留) 主程式：循環計算與繪圖 ---
# (此函式無需修改)
def main():
    
    # --- (可修改) 參數設定 ---
    EXPERIMENT_TYPE = ['time_variation_1', 'time_variation_2'][1]  # 選擇 'time_variation' 或 'spatial_variation'
    
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [0.1, 1.0, 10.0]
    BETAS = [0.1, 1.0, 10.0]
    EPOCH = 1
    
    if EXPERIMENT_TYPE == 'time_variation_1':
        SOURCE_FILE_TAG = '190611' # 假設 source tag 不變
        TARGET_FILE_TAG = '191009' # 假設 target tag 不變
        print(f"開始分析 [time_variation_1] 實驗 (Source: {SOURCE_FILE_TAG}, Target: {TARGET_FILE_TAG})")
    elif EXPERIMENT_TYPE == 'time_variation_2':
        SOURCE_FILE_TAG = '190611' # 假設 source tag 不變
        TARGET_FILE_TAG = '200219' # 假設 target tag 不變
        print(f"開始分析 [time_variation_2] 實驗 (Source: {SOURCE_FILE_TAG}, Target: {TARGET_FILE_TAG})")
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

    if not all_results:
        print("錯誤：沒有收集到任何結果。")
        return
        
    results_df = pd.DataFrame(all_results)
    
    print("\n--- 完整 MDE 結果 (NaN 代表檔案缺失) ---")
    print(results_df)

    # 繪製兩張圖
    plot_mde_chart(results_df, 'labeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_labeled_mde_results_NewData.png')
    plot_mde_chart(results_df, 'unlabeled', EXPERIMENT_TYPE, f'{EXPERIMENT_TYPE}_unlabeled_mde_results_NewData.png')


if __name__ == "__main__":
    main()