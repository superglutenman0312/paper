import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 核心函式 (座標轉換與距離計算 - UM_DSI_DB 專用) ---

# 新資料集的「標籤/座標 對照表」 (字典)
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
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        raise KeyError(f"標籤 {a} 在 LABEL_TO_COORDINATE_DICT 中找不到。")

def euclidean_distance(p1, p2):
    # 此資料集使用的是標準歐幾里得距離 (無縮放因子)
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_mde_from_file(file_path):
    """
    讀取指定的預測 CSV 檔案，並計算 MDE。
    """
    if not os.path.exists(file_path):
        return np.nan 

    try:
        results = pd.read_csv(file_path)
        if 'label' not in results.columns or 'pred' not in results.columns:
            return np.nan
    except Exception:
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
        except Exception:
            continue

    if errors:
        return np.mean(errors)
    else:
        return np.nan


# --- 2. 繪圖函式 (單一種子繪圖) ---

def plot_single_seed_chart(df, mode, seed, experiment_name, output_filename):
    """
    繪製單一 Random Seed 的長條圖
    """
    mode_df = df[df['mode'] == mode].copy()
    if mode_df.empty:
        print(f"  [Skip] Seed {seed} - {mode}: 沒有資料可供繪圖。")
        return

    # 確保按照 Beta 大小排序
    mode_df = mode_df.sort_values(by=['beta'])

    combo_labels = mode_df['combo_label'].unique()
    source_mdes = mode_df[mode_df['type'] == 'Source']['mde'].values
    target_mdes = mode_df[mode_df['type'] == 'Target']['mde'].values

    x = np.arange(len(combo_labels))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(14, 8)) 
    
    rects1 = ax.bar(x - width/2, source_mdes, width, label='Source MDE', alpha=0.9)
    rects2 = ax.bar(x + width/2, target_mdes, width, label='Target MDE', alpha=0.9)

    ax.set_ylabel('Mean Distance Error (MDE) [m]')
    ax.set_xlabel('Loss Weights Parameters')
    ax.set_title(f'MDE Analysis for {mode.capitalize()} Models\nExperiment: {experiment_name} (Seed {seed})')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  成功儲存圖表: {output_filename}")


# --- 3. 主程式：循環計算與繪圖 ---

def main():
    
    # --- 參數設定 ---
    EXPERIMENT_TYPES = ['time_variation_1', 'time_variation_2']
    
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [1.0] 
    BETAS = [1.0, 10.0, 100.0]
    EPOCH = 100
    RANDOM_SEEDS = [42, 70, 100] 
    # BETAS = [0.1, 1.0]
    # EPOCH = 1
    # RANDOM_SEEDS = [42, 70] 
    
    # --- 最外層：遍歷實驗類型 ---
    for exp_type in EXPERIMENT_TYPES:
        print(f"\n{'#'*60}")
        print(f"開始分析實驗類型: {exp_type}")
        print(f"{'#'*60}")

        # 設定檔案標籤
        if exp_type == 'time_variation_1':
            SOURCE_FILE_TAG = '190611'
            TARGET_FILE_TAG = '191009'
        elif exp_type == 'time_variation_2':
            SOURCE_FILE_TAG = '190611'
            TARGET_FILE_TAG = '200219'
        else:
            continue

        base_work_dir = exp_type

        # --- 中層：遍歷 Random Seed ---
        for seed in RANDOM_SEEDS:
            print(f"\n--- 正在處理 Random Seed: {seed} ({exp_type}) ---")
            
            seed_base_dir = os.path.join(base_work_dir, f'random_seed_{seed}')
            if not os.path.exists(seed_base_dir):
                print(f"警告: 找不到資料夾 {seed_base_dir}，跳過。")
                continue
            
            seed_results = []

            for mode in MODES:
                for alpha in ALPHAS:
                    for beta in BETAS:
                        
                        folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH}_{mode}"
                        model_dir = os.path.join(seed_base_dir, folder_name, 'predictions')
                        
                        combo_label = f"α={alpha}\nβ={beta}"

                        source_path = os.path.join(model_dir, f'{SOURCE_FILE_TAG}_results.csv')
                        target_path = os.path.join(model_dir, f'{TARGET_FILE_TAG}_results.csv')
                        
                        mde_source = calculate_mde_from_file(source_path)
                        mde_target = calculate_mde_from_file(target_path)

                        if not np.isnan(mde_source) and not np.isnan(mde_target):
                            seed_results.append({ 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Source', 'mde': mde_source })
                            seed_results.append({ 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Target', 'mde': mde_target })

            if not seed_results:
                print(f"Seed {seed} ({exp_type}) 沒有收集到任何有效數據。")
                continue
                
            seed_df = pd.DataFrame(seed_results)
            
            # 繪圖並存檔 (存在該 Seed 的資料夾內)
            # 1. Labeled
            labeled_out_path = os.path.join(seed_base_dir, f'seed_{seed}_{exp_type}_labeled_mde.png')
            plot_single_seed_chart(seed_df, 'labeled', seed, exp_type, labeled_out_path)
            
            # 2. Unlabeled
            unlabeled_out_path = os.path.join(seed_base_dir, f'seed_{seed}_{exp_type}_unlabeled_mde.png')
            plot_single_seed_chart(seed_df, 'unlabeled', seed, exp_type, unlabeled_out_path)

    print("\n所有實驗分析完成！")

if __name__ == "__main__":
    main()