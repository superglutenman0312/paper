import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 參數與設定區 =================

# 1. 隨機種子與 Epoch 設定
SEEDS = [42, 70, 100]
EPOCH_SWD = 100
EPOCH_WD = 100

# # 2. 定義要比較的 WD 參數變體
# WD_VARIANTS = [
#     (1.0, 1.0, "WD_1_1"),
#     (1.0, 10.0, "WD_1_10"),
#     (1.0, 100.0, "WD_1_100")
# ]

# SWD_VARIANTS = [
#     (0.0, 1.0, "SWD_0_1"),
#     (1.0, 1.0, "SWD_1_1"),
#     (1.0, 5.0, "SWD_1_5"),
#     (1.0, 10.0, "SWD_1_10"),
#     (1.0, 25.0, "SWD_1_25"),
#     (1.0, 50.0, "SWD_1_50"),
#     (1.0, 100.0, "SWD_1_100")
# ]

SWD_VARIANTS = [
    (1.0, 1.0, "SWD_1_1"),
    (1.0, 5.0, "SWD_1_5"),
    (1.0, 10.0, "SWD_1_10"),
    (1.0, 25.0, "SWD_1_25"),
    (1.0, 50.0, "SWD_1_50"),
    (1.0, 100.0, "SWD_1_100")
]

# 4. 定義其他方法 (將 DNN 移至最前)
OTHER_METHODS = ["DNN", "DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 5. 定義實驗場景
SCENARIOS = {
    "experiments2": ("200925_results.csv", "211204_results.csv")
}

# 6. 輸出目錄
OUTPUT_DIR = "Comparison_Result2"

# 7. 根目錄名稱
SWD_ROOT_NAME = "SWD"
WD_ROOT_NAME = "WD"

# 【新增】Label Map 檔案路徑 (請依實際位置修改)
MAP_FILE_PATH = r'D:/paper_thesis/My/data/MTLocData/OfficeP1+P2/processed_data/OfficeP1+P2_1_training_vs_OfficeP1+P2_38_training/experiment_class_map.csv'

# ================= 核心函式 (座標與距離) =================

# 【修改】不再 Hardcode，初始化為空，改由 load_coordinate_map 讀取
LABEL_TO_COORDINATE_DICT = {}

def load_coordinate_map(filepath):
    """讀取 experiment_class_map.csv 並更新全域字典"""
    global LABEL_TO_COORDINATE_DICT
    
    if not os.path.exists(filepath):
        print(f"[Error] 找不到 Map 檔案: {filepath}")
        return
    
    try:
        df = pd.read_csv(filepath)
        mapping = {}
        # 讀取 class_id, x, y
        for _, row in df.iterrows():
            cid = int(row['class_id'])
            x = float(row['x'])
            y = float(row['y'])
            mapping[cid] = (x, y)
            
        LABEL_TO_COORDINATE_DICT = mapping
        print(f"成功載入 Map: {filepath} (共 {len(mapping)} 個地點)")
    except Exception as e:
        print(f"[Error] 讀取 Map 失敗: {e}")

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

# === 新增函式: 計算並回傳誤差列表 (用於 CDF) ===
def calculate_errors_from_file(file_path):
    """
    與 calculate_mde_from_file 邏輯一致，但回傳完整的 errors list
    """
    if not os.path.exists(file_path):
        return []

    try:
        results = pd.read_csv(file_path)
        if 'label' not in results.columns or 'pred' not in results.columns:
            return []
    except Exception:
        return []

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

    return errors


# ================= 資料讀取邏輯 =================

# def get_wd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
#     ... (保留原始註解與程式碼)

def get_swd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
    """
    取得 SWD 的 MDE
    """
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    
    mde_s = calculate_mde_from_file(os.path.join(model_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(model_dir, tgt_file))
    
    return mde_s, mde_t

def get_other_mde(exp_type, seed, mode, method_name, src_file, tgt_file):
    """
    取得其他方法的 MDE (已針對 DNN 做路徑特例處理)
    """
    base_path = os.path.join(method_name, exp_type, f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return np.nan, np.nan
    
    # === 修改部分: 針對 DNN 進行路徑特判 ===
    if method_name == "DNN":
        # DNN 的資料夾通常不分 labeled/unlabeled，且名稱含 "SourceOnly"
        # 根據目錄結構，資料夾位於 .../time_reversal_1/random_seed_42/SourceOnly_Epoch1
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


# === 新增函式: 取得原始誤差列表 (Target Domain) ===

def get_swd_errors(exp_type, seed, mode, alpha, beta, tgt_file):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    return calculate_errors_from_file(os.path.join(model_dir, tgt_file))

def get_other_errors(exp_type, seed, mode, method_name, tgt_file):
    base_path = os.path.join(method_name, exp_type, f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return []
    
    if method_name == "DNN":
        search_pattern = os.path.join(base_path, "*SourceOnly*")
    else:
        search_pattern = os.path.join(base_path, f"*{mode}")

    found_dirs = glob.glob(search_pattern)
    if not found_dirs:
        return []
        
    pred_dir = os.path.join(found_dirs[0], "predictions")
    return calculate_errors_from_file(os.path.join(pred_dir, tgt_file))


# ================= 繪圖函式 =================

def plot_combined_comparison(df, title_seed_part, exp_name, output_filename):
    """
    畫圖函式
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
            
        ordered_names = [v[2] for v in SWD_VARIANTS] + \
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


# === 新增繪圖函式: 繪製 CDF ===
def plot_cdf_comparison(data_dict, title_seed_part, exp_name, output_filename):
    """
    繪製 CDF 比較圖 (Target Domain)
    data_dict 結構: data_dict[mode][method_name] = [err1, err2, ...]
    """
    modes = ['labeled', 'unlabeled']
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f'CDF Comparison (Target) - {title_seed_part}\nExperiment: {exp_name}', fontsize=16)
    
    # 預定義顏色或樣式循環，避免太多線混淆
    colors = plt.cm.tab20.colors  # 使用有20種顏色的 colormap
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        
        if mode not in data_dict or not data_dict[mode]:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            ax.set_title(f'{mode.capitalize()} Mode')
            continue
            
        # 依照固定順序繪圖
        ordered_names = [v[2] for v in SWD_VARIANTS] + OTHER_METHODS
        
        has_data = False
        color_idx = 0
        
        for name in ordered_names:
            if name in data_dict[mode] and len(data_dict[mode][name]) > 0:
                errors = np.array(data_dict[mode][name])
                sorted_errors = np.sort(errors)
                y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                
                # 繪線
                color = colors[color_idx % len(colors)]
                ax.plot(sorted_errors, y_vals, label=name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
                has_data = True
        
        ax.set_ylabel('CDF (Probability)')
        ax.set_xlabel('Distance Error (m)')
        ax.set_title(f'{mode.capitalize()} Mode')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize='small')
        
        # 設定 x 軸範圍，避免 outliers 讓圖太扁，取 98百分位數或固定值
        # ax.set_xlim(0, 20) # 可視需要固定
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Valid Error Data', ha='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> CDF 圖表已儲存: {output_filename}")


# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 【新增】程式開始時先載入 Map
    load_coordinate_map(MAP_FILE_PATH)

    for exp_type, (src_file, tgt_file) in SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"開始處理場景: {exp_type}")
        print(f"Source: {src_file}, Target: {tgt_file}")
        print(f"{'='*50}")
        
        all_seeds_data = [] 
        
        # === 新增: 用於儲存所有 seed 的誤差數據以畫 Aggregate CDF ===
        # 結構: cdf_global_data[mode][method_name] = [err1, err2, ...] (累加所有 seed)
        cdf_global_data = {'labeled': {}, 'unlabeled': {}} 

        for seed in SEEDS:
            print(f"Processing Random Seed: {seed}")
            current_seed_data = []
            
            # 暫存當下 seed 的誤差數據 (如果想畫單一 seed 的 CDF)
            cdf_seed_data = {'labeled': {}, 'unlabeled': {}}
            
            for mode in ['labeled', 'unlabeled']:
                
                # --- A. 讀取 WD --- (略，保留原始註解)

                # --- B. 讀取 SWD ---
                for (alpha, beta, name) in SWD_VARIANTS:
                    mde_s, mde_t = get_swd_mde(exp_type, seed, mode, alpha, beta, src_file, tgt_file)
                    
                    if not np.isnan(mde_s) and not np.isnan(mde_t):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Target', 'mde': mde_t})
                    
                    # === 新增: 收集 Target Errors 用於 CDF ===
                    errs = get_swd_errors(exp_type, seed, mode, alpha, beta, tgt_file)
                    if errs:
                        # 存入 global (所有 seed)
                        if name not in cdf_global_data[mode]: cdf_global_data[mode][name] = []
                        cdf_global_data[mode][name].extend(errs)
                        # 存入 local (當前 seed)
                        if name not in cdf_seed_data[mode]: cdf_seed_data[mode][name] = []
                        cdf_seed_data[mode][name].extend(errs)

                # --- C. 讀取其他方法 (含 DNN) ---
                for method in OTHER_METHODS:
                    mde_s, mde_t = get_other_mde(exp_type, seed, mode, method, src_file, tgt_file)
                    
                    if not np.isnan(mde_s) and not np.isnan(mde_t):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Target', 'mde': mde_t})
                    
                    # === 新增: 收集 Target Errors 用於 CDF ===
                    errs = get_other_errors(exp_type, seed, mode, method, tgt_file)
                    if errs:
                        if method not in cdf_global_data[mode]: cdf_global_data[mode][method] = []
                        cdf_global_data[mode][method].extend(errs)
                        if method not in cdf_seed_data[mode]: cdf_seed_data[mode][method] = []
                        cdf_seed_data[mode][method].extend(errs)

            if current_seed_data:
                df_seed = pd.DataFrame(current_seed_data)
                out_name = os.path.join(OUTPUT_DIR, f"{exp_type}_seed_{seed}.png")
                plot_combined_comparison(df_seed, f"Random Seed {seed}", exp_type, out_name)
                
                # === 新增: 繪製該 Seed 的 CDF ===
                out_name_cdf = os.path.join(OUTPUT_DIR, f"{exp_type}_seed_{seed}_CDF.png")
                plot_cdf_comparison(cdf_seed_data, f"Random Seed {seed}", exp_type, out_name_cdf)
                
                all_seeds_data.extend(current_seed_data)
            else:
                print(f"  [Warn] Seed {seed} 沒有有效資料。")

        if all_seeds_data:
            print(f"\n正在計算 {exp_type} 的平均結果...")
            df_all = pd.DataFrame(all_seeds_data)
            df_avg = df_all.groupby(['mode', 'method_name', 'type'], as_index=False)['mde'].mean()
            
            out_name_avg = os.path.join(OUTPUT_DIR, f"{exp_type}_average.png")
            plot_combined_comparison(df_avg, "Average Result", exp_type, out_name_avg)
            
            # === 新增: 繪製 Average (Combined Seeds) CDF ===
            out_name_avg_cdf = os.path.join(OUTPUT_DIR, f"{exp_type}_average_CDF.png")
            plot_cdf_comparison(cdf_global_data, "Average Result (All Seeds Pooled)", exp_type, out_name_avg_cdf)
        else:
            print(f"  [Error] {exp_type} 完全沒有資料。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()