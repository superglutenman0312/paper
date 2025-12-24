import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 參數與設定區 =================

# 1. 隨機種子與 Epoch 設定
SEEDS = [42, 70, 100]
EPOCH_SWD = 500
EPOCH_WD = 500

# 2. 定義要比較的 WD 參數變體
WD_VARIANTS = [
    (1.0, 1.0, "WD_1_1"),
    (1.0, 10.0, "WD_1_10"),
    (1.0, 100.0, "WD_1_100")
]

# 3. 定義要比較的 SWD 參數變體
SWD_VARIANTS = [
    (0.0, 1.0, "SWD_0_1"),
    (1.0, 1.0, "SWD_1_1"),
    (1.0, 5.0, "SWD_1_5"),
    (1.0, 10.0, "SWD_1_10"),
    (1.0, 25.0, "SWD_1_25"),
    (1.0, 50.0, "SWD_1_50"),
    (1.0, 100.0, "SWD_1_100")
]

# 4. 定義方法
DNN_METHOD = "DNN"
OTHER_METHODS = ["DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 5. 定義實驗場景設定
SCENARIO_CONFIG = {
    "time_variation": { 
        "other_folder": "time_variation", 
        "files": ("220318_results.csv", "231116_results.csv") 
    },
    "spatial_variation": { 
        "other_folder": "spatial_variation", 
        "files": ("231116_results.csv", "231117_results.csv") 
    }
}

# 6. 輸出目錄
OUTPUT_DIR = "Comparison_Result3"

# 7. 根目錄名稱
SWD_ROOT_NAME = "SWD"
WD_ROOT_NAME = "WD"

# 【新增】Label Map 檔案路徑 (請修改為您的實際路徑)
MAP_FILE_PATH = r"D:\paper_thesis\My\data\MCSL\processed_data\20220318_20231116\experiment_class_map.csv"
# MAP_FILE_PATH = r"D:\paper_thesis\My\data\MCSL\processed_data\20231116_20231117\experiment_class_map.csv"

# ================= 核心函式 (MCSL 座標與距離) =================

# 【修改】初始化為空，改由 load_coordinate_map 讀取
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
    # 【修改】計算歐氏距離後乘以 0.6 (比例尺)
    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance * 0.6

def calculate_mde_from_file(file_path):
    if not os.path.exists(file_path):
        return np.nan 

    try:
        results = pd.read_csv(file_path)
        if 'label' not in results.columns or 'pred' not in results.columns:
            return np.nan
        
        results['label'] = pd.to_numeric(results['label'], errors='coerce').fillna(0).astype(int)
        results['pred'] = pd.to_numeric(results['pred'], errors='coerce').fillna(0).astype(int)

    except Exception:
        return np.nan

    errors = []
    for idx, row in results.iterrows():
        try:
            pred_label = int(row['pred'])
            actual_label = int(row['label'])
            
            if pred_label == 0 or actual_label == 0:
                continue

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

# === 【新增】函式: 計算並回傳誤差列表 (用於 CDF) ===
def calculate_errors_from_file(file_path):
    """與 calculate_mde_from_file 邏輯一致，但回傳完整的 errors list"""
    if not os.path.exists(file_path):
        return []

    try:
        results = pd.read_csv(file_path)
        if 'label' not in results.columns or 'pred' not in results.columns:
            return []
        results['label'] = pd.to_numeric(results['label'], errors='coerce').fillna(0).astype(int)
        results['pred'] = pd.to_numeric(results['pred'], errors='coerce').fillna(0).astype(int)
    except Exception:
        return []

    errors = []
    for idx, row in results.iterrows():
        try:
            pred_label = int(row['pred'])
            actual_label = int(row['label'])
            if pred_label == 0 or actual_label == 0: continue
            pred_coord = class_to_coordinate(pred_label)
            actual_coord = class_to_coordinate(actual_label)
            if pred_coord is None or actual_coord is None: continue
            distance_error = euclidean_distance(pred_coord, actual_coord)
            errors.append(distance_error)
        except Exception:
            continue
    return errors

# ================= 資料讀取邏輯 =================

def get_wd_mde_and_errors(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_WD}_{mode}"
    model_dir = os.path.join(WD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    mde_s = calculate_mde_from_file(os.path.join(model_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(model_dir, tgt_file))
    errs_t = calculate_errors_from_file(os.path.join(model_dir, tgt_file))
    return mde_s, mde_t, errs_t

def get_swd_mde_and_errors(exp_type, seed, mode, alpha, beta, src_file, tgt_file):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_ROOT_NAME, exp_type, f'random_seed_{seed}', folder_name, 'predictions')
    mde_s = calculate_mde_from_file(os.path.join(model_dir, src_file))
    mde_t = calculate_mde_from_file(os.path.join(model_dir, tgt_file))
    errs_t = calculate_errors_from_file(os.path.join(model_dir, tgt_file))
    return mde_s, mde_t, errs_t

def get_other_mde_and_errors(exp_type, other_folder_name, seed, mode, method_name, src_file, tgt_file):
    base_path = os.path.join(method_name, other_folder_name, f"random_seed_{seed}")
    if not os.path.exists(base_path): return np.nan, np.nan, []
    
    if method_name == DNN_METHOD:
        search_pattern = os.path.join(base_path, "*SourceOnly*")
    else:
        search_pattern = os.path.join(base_path, f"*{mode}")
    
    found_dirs = glob.glob(search_pattern)
    if not found_dirs: return np.nan, np.nan, []
        
    pred_dir = os.path.join(found_dirs[0], "predictions")
    path_s = os.path.join(pred_dir, src_file)
    path_t = os.path.join(pred_dir, tgt_file)
    
    mde_s = calculate_mde_from_file(path_s)
    mde_t = calculate_mde_from_file(path_t)
    errs_t = calculate_errors_from_file(path_t)
    
    return mde_s, mde_t, errs_t

# ================= 繪圖函式 =================

def plot_combined_comparison(df, title_seed_part, exp_name, output_filename):
    modes = ['labeled', 'unlabeled']
    fig, axes = plt.subplots(1, 2, figsize=(22, 8)) 
    fig.suptitle(f'MDE Comparison (MCSL) - {title_seed_part}\nExperiment: {exp_name}', fontsize=16)
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
        ordered_names = [v[2] for v in WD_VARIANTS] + [v[2] for v in SWD_VARIANTS] + [DNN_METHOD] + OTHER_METHODS
        plot_names, source_vals, target_vals = [], [], []
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
        ax.set_xticklabels(plot_names, rotation=45, ha='right')
        ax.set_ylim(0, global_max * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if i == 0: ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> MDE 圖表已儲存: {output_filename}")

# === 【新增】繪圖函式: 繪製 CDF ===
def plot_cdf_comparison(data_dict, title_seed_part, exp_name, output_filename):
    """繪製 CDF 比較圖 (Target Domain)"""
    modes = ['labeled', 'unlabeled']
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f'CDF Comparison (Target) - {title_seed_part}\nExperiment: {exp_name}', fontsize=16)
    colors = plt.cm.tab20.colors
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        if mode not in data_dict or not data_dict[mode]:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            ax.set_title(f'{mode.capitalize()} Mode')
            continue
        ordered_names = [v[2] for v in WD_VARIANTS] + [v[2] for v in SWD_VARIANTS] + [DNN_METHOD] + OTHER_METHODS
        has_data = False
        color_idx = 0
        for name in ordered_names:
            if name in data_dict[mode] and len(data_dict[mode][name]) > 0:
                errors = np.array(data_dict[mode][name])
                sorted_errors = np.sort(errors)
                y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                color = colors[color_idx % len(colors)]
                ax.plot(sorted_errors, y_vals, label=name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
                has_data = True
        ax.set_ylabel('CDF (Probability)')
        ax.set_xlabel('Distance Error (m)')
        ax.set_title(f'{mode.capitalize()} Mode')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize='small')
        if not has_data: ax.text(0.5, 0.5, 'No Valid Error Data', ha='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> CDF 圖表已儲存: {output_filename}")

# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"建立輸出目錄: {OUTPUT_DIR}")

    # 【新增】程式開始時先載入 Map
    load_coordinate_map(MAP_FILE_PATH)

    for swd_exp_type, config in SCENARIO_CONFIG.items():
        other_exp_folder = config["other_folder"]
        src_file, tgt_file = config["files"]
        print(f"\n{'='*60}\n處理場景: {swd_exp_type}\n  - Files: {src_file}, {tgt_file}\n{'='*60}")
        
        all_seeds_data = [] 
        # 【新增】用於儲存所有 seed 的誤差數據以畫 Aggregate CDF
        cdf_global_data = {'labeled': {}, 'unlabeled': {}} 

        for seed in SEEDS:
            print(f"Processing Random Seed: {seed}")
            current_seed_data = []
            # 【新增】暫存當下 seed 的誤差數據
            cdf_seed_data = {'labeled': {}, 'unlabeled': {}}
            
            for mode in ['labeled', 'unlabeled']:
                # --- A. 讀取 WD ---
                for (alpha, beta, name) in WD_VARIANTS:
                    mde_s, mde_t, errs_t = get_wd_mde_and_errors(swd_exp_type, seed, mode, alpha, beta, src_file, tgt_file)
                    if not np.isnan(mde_s):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Target', 'mde': mde_t})
                    if errs_t:
                        if name not in cdf_global_data[mode]: cdf_global_data[mode][name] = []
                        cdf_global_data[mode][name].extend(errs_t)
                        if name not in cdf_seed_data[mode]: cdf_seed_data[mode][name] = []
                        cdf_seed_data[mode][name].extend(errs_t)

                # --- B. 讀取 SWD ---
                for (alpha, beta, name) in SWD_VARIANTS:
                    mde_s, mde_t, errs_t = get_swd_mde_and_errors(swd_exp_type, seed, mode, alpha, beta, src_file, tgt_file)
                    if not np.isnan(mde_s):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': name, 'type': 'Target', 'mde': mde_t})
                    if errs_t:
                        if name not in cdf_global_data[mode]: cdf_global_data[mode][name] = []
                        cdf_global_data[mode][name].extend(errs_t)
                        if name not in cdf_seed_data[mode]: cdf_seed_data[mode][name] = []
                        cdf_seed_data[mode][name].extend(errs_t)

                # --- C. 讀取 DNN 與 其他方法 ---
                ALL_COMPARED_METHODS = [DNN_METHOD] + OTHER_METHODS
                for method in ALL_COMPARED_METHODS:
                    mde_s, mde_t, errs_t = get_other_mde_and_errors(swd_exp_type, other_exp_folder, seed, mode, method, src_file, tgt_file)
                    if not np.isnan(mde_s):
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Source', 'mde': mde_s})
                        current_seed_data.append({'seed': seed, 'mode': mode, 'method_name': method, 'type': 'Target', 'mde': mde_t})
                    if errs_t:
                        if method not in cdf_global_data[mode]: cdf_global_data[mode][method] = []
                        cdf_global_data[mode][method].extend(errs_t)
                        if method not in cdf_seed_data[mode]: cdf_seed_data[mode][method] = []
                        cdf_seed_data[mode][method].extend(errs_t)
            
            if current_seed_data:
                df_seed = pd.DataFrame(current_seed_data)
                out_name = os.path.join(OUTPUT_DIR, f"{swd_exp_type}_seed_{seed}.png")
                plot_combined_comparison(df_seed, f"Random Seed {seed}", swd_exp_type, out_name)
                # 【新增】繪製該 Seed 的 CDF
                out_name_cdf = os.path.join(OUTPUT_DIR, f"{swd_exp_type}_seed_{seed}_CDF.png")
                plot_cdf_comparison(cdf_seed_data, f"Random Seed {seed}", swd_exp_type, out_name_cdf)
                all_seeds_data.extend(current_seed_data)
            else:
                print(f"  [Warn] Seed {seed} 沒有有效資料。")

        if all_seeds_data:
            print(f"\n正在計算 {swd_exp_type} 的平均結果...")
            df_all = pd.DataFrame(all_seeds_data)
            df_avg = df_all.groupby(['mode', 'method_name', 'type'], as_index=False)['mde'].mean()
            out_name_avg = os.path.join(OUTPUT_DIR, f"{swd_exp_type}_average.png")
            plot_combined_comparison(df_avg, "Average Result", swd_exp_type, out_name_avg)
            # 【新增】繪製 Average CDF
            out_name_avg_cdf = os.path.join(OUTPUT_DIR, f"{swd_exp_type}_average_CDF.png")
            plot_cdf_comparison(cdf_global_data, "Average Result (All Seeds Pooled)", swd_exp_type, out_name_avg_cdf)
        else:
            print(f"  [Error] {swd_exp_type} 完全沒有資料。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()