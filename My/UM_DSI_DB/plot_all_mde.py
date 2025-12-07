import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 參數與設定區 =================

# 1. 隨機種子與 Epoch 設定
SEEDS = [42, 70, 100]
EPOCH_SWD = 100

# 2. 定義要比較的 WD 參數變體
WD_VARIANTS = [
    (1.0, 1.0, "WD_1_1"),
    (1.0, 10.0, "WD_1_10"),
    (1.0, 100.0, "WD_1_100")
]

SWD_VARIANTS = [
    (1.0, 1.0, "SWD_1_1"),
    (1.0, 5.0, "SWD_1_5"),
    (1.0, 10.0, "SWD_1_10"),
    (1.0, 25.0, "SWD_1_25"),
    (1.0, 50.0, "SWD_1_50"),
    (1.0, 100.0, "SWD_1_100")
]

# 4. 定義其他方法 (DNN 已移至最前)
OTHER_METHODS = ["DNN", "DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 5. 定義實驗場景
SCENARIOS = {
    "time_variation_1": ("190611_results.csv", "191009_results.csv"),
    "time_variation_2": ("190611_results.csv", "200219_results.csv")
}

# 6. 輸出目錄
OUTPUT_DIR = "Comparison_Result2"

# 7. 根目錄名稱設定
SWD_ROOT_NAME = "SWD"
WD_ROOT_NAME = "WD"

# ================= 核心函式 (座標與距離) =================

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
        return None

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_mde_from_file(file_path):
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
            
            if pred_coord is None or actual_coord is None:
                continue

            distance_error = euclidean_distance(pred_coord, actual_coord)
            errors.append(distance_error)
        except Exception:
            continue

    if errors:
        return np.mean(errors)
    else:
        return np.nan


# ================= 資料讀取邏輯 =================

def get_wd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(WD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    
    mde_s = calculate_mde_from_file(os.path.join(model_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(model_dir, tgt_file))
    
    return mde_s, mde_t

def get_swd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    
    mde_s = calculate_mde_from_file(os.path.join(model_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(model_dir, tgt_file))
    
    return mde_s, mde_t

def get_other_mde(exp_type, seed, mode, method_name, src_file, tgt_file):
    """
    取得其他方法的 MDE (已加入 DNN 特例處理)
    """
    base_path = os.path.join(method_name, exp_type, f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return np.nan, np.nan
    
    # === 修改部分: 針對 DNN 進行路徑特判 ===
    if method_name == "DNN":
        # DNN 的資料夾通常不分 labeled/unlabeled，且名稱含 "SourceOnly"
        # 根據你的截圖，資料夾位於 .../time_variation_1/random_seed_42/SourceOnly_Epoch1
        search_pattern = os.path.join(base_path, "*SourceOnly*")
    else:
        # 其他方法 (DANN 等) 依賴 labeled/unlabeled 後綴
        search_pattern = os.path.join(base_path, f"*{mode}")
    # =======================================
    
    found_dirs = glob.glob(search_pattern)
    
    if not found_dirs:
        return np.nan, np.nan
        
    pred_dir = os.path.join(found_dirs[0], "predictions")
    
    mde_s = calculate_mde_from_file(os.path.join(pred_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(pred_dir, tgt_file))
    
    return mde_s, mde_t


# ================= 繪圖函式 =================

def plot_combined_comparison(df, title_seed_part, exp_name, output_filename):
    """
    畫圖函式：左 Labeled, 右 Unlabeled
    """
    modes = ['labeled', 'unlabeled']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f'MDE Comparison - {title_seed_part}\nExperiment: {exp_name}', fontsize=16)
    
    color_source = '#4c72b0' 
    color_target = '#dd8452'
    
    global_max = df['mde'].max() if not df.empty else 1.0

    for i, mode in enumerate(modes):
        ax = axes[i]
        mode_df = df[df['mode'] == mode].copy()
        
        if mode_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            ax.set_title(f'{mode.capitalize()} Mode')
            continue
            
        # 排序：WD -> SWD -> Other Methods (DNN, DANN...)
        ordered_names = [v[2] for v in WD_VARIANTS] + \
                        [v[2] for v in SWD_VARIANTS] + \
                        OTHER_METHODS
        
        plot_names = []
        source_vals = []
        target_vals = []
        
        for name in ordered_names:
            subset = mode_df[mode_df['method_name'] == name]
            if not subset.empty:
                val_s = subset[subset['type'] == 'Source']['mde'].values
                val_t = subset[subset['type'] == 'Target']['mde'].values
                
                s_score = val_s[0] if len(val_s) > 0 else 0
                t_score = val_t[0] if len(val_t) > 0 else 0
                
                plot_names.append(name)
                source_vals.append(s_score)
                target_vals.append(t_score)

        x = np.arange(len(plot_names))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, source_vals, width, label='Source', color=color_source, alpha=0.9)
        rects2 = ax.bar(x + width/2, target_vals, width, label='Target', color=color_target, alpha=0.9)
        
        ax.set_ylabel('MDE (m)')
        ax.set_title(f'{mode.capitalize()} Mode')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_names, rotation=35, ha='right')
        
        ax.set_ylim(0, global_max * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend()
            
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> 圖表已儲存: {output_filename}")


# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for exp_type, (src_file, tgt_file) in SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"開始處理場景: {exp_type}")
        print(f"Source: {src_file}, Target: {tgt_file}")
        print(f"{'='*50}")
        
        all_seeds_data = []

        for seed in SEEDS:
            print(f"Processing Random Seed: {seed}")
            current_seed_data = []
            
            for mode in ['labeled', 'unlabeled']:
                
                # --- A. 讀取 WD ---
                for (alpha, beta, name) in WD_VARIANTS:
                    mde_s, mde_t = get_wd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file)
                    if not np.isnan(mde_s) and not np.isnan(mde_t):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Target', 'mde': mde_t})

                # --- B. 讀取 SWD ---
                for (alpha, beta, name) in SWD_VARIANTS:
                    mde_s, mde_t = get_swd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file)
                    if not np.isnan(mde_s) and not np.isnan(mde_t):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Target', 'mde': mde_t})

                # --- C. 讀取 其他方法 (含 DNN) ---
                for method in OTHER_METHODS:
                    mde_s, mde_t = get_other_mde(exp_type, seed, mode, method, src_file, tgt_file)
                    if not np.isnan(mde_s) and not np.isnan(mde_t):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Target', 'mde': mde_t})
            
            if current_seed_data:
                df_seed = pd.DataFrame(current_seed_data)
                out_name = os.path.join(OUTPUT_DIR, f"{exp_type}_seed_{seed}.png")
                plot_combined_comparison(df_seed, f"Random Seed {seed}", exp_type, out_name)
                all_seeds_data.extend(current_seed_data)
            else:
                print(f"  [Warn] Seed {seed} 沒有有效資料。")

        if all_seeds_data:
            print(f"\n正在計算 {exp_type} 的平均結果...")
            df_all = pd.DataFrame(all_seeds_data)
            df_avg = df_all.groupby(['mode', 'method_name', 'type'], as_index=False)['mde'].mean()
            out_name_avg = os.path.join(OUTPUT_DIR, f"{exp_type}_average.png")
            plot_combined_comparison(df_avg, "Average Result", exp_type, out_name_avg)
        else:
            print(f"  [Error] {exp_type} 完全沒有資料。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()