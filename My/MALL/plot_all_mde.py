import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle

# ================= 參數設定區 =================

# 1. 隨機與實驗設定
SEEDS = [42, 70, 100]
EPOCH_SWD = 100

# 2. 定義要比較的 WD 參數
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

# 4. 定義其他方法 (DNN 已包含在內)
OTHER_METHODS = ["DNN", "DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 5. 檔案標籤 (Source / Target)
FILE_SOURCE = "211120_results.csv"
FILE_TARGET = "221221_results.csv"

# 6. Label Map 路徑
LABEL_MAP_PATH = 'D:/paper_thesis/My/data/MTLocData/Mall/label_map.pkl'

# 7. 路徑設定
SWD_BASE_DIR = os.path.join("SWD", "experiments")
WD_BASE_DIR = os.path.join("WD", "experiments")
OUTPUT_DIR = "Comparison_Result2"

# ================= 工具函式 =================

# --- 1. 載入 Label Map ---
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    LABEL_TO_COORDINATE_DICT = {value: key for key, value in label_map.items()}
    print(f"成功載入座標對照表，共 {len(LABEL_TO_COORDINATE_DICT)} 筆資料。")
except Exception as e:
    print(f"[錯誤] 無法載入 Label Map: {e}")
    LABEL_TO_COORDINATE_DICT = {}

def class_to_coordinate(a):
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        return None

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- 2. 計算 MDE ---
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

# --- 3. 取得資料路徑的邏輯 ---

def get_swd_mde(seed, mode, alpha, beta):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_BASE_DIR, f"random_seed_{seed}", folder_name, "predictions")
    
    src_path = os.path.join(model_dir, FILE_SOURCE)
    tgt_path = os.path.join(model_dir, FILE_TARGET)
    
    mde_s = calculate_mde_from_file(src_path)
    mde_t = calculate_mde_from_file(tgt_path)
    
    return mde_s, mde_t

def get_wd_mde(seed, mode, alpha, beta):
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(WD_BASE_DIR, f"random_seed_{seed}", folder_name, "predictions")
    
    src_path = os.path.join(model_dir, FILE_SOURCE)
    tgt_path = os.path.join(model_dir, FILE_TARGET)
    
    mde_s = calculate_mde_from_file(src_path)
    mde_t = calculate_mde_from_file(tgt_path)
    
    return mde_s, mde_t

def get_other_mde(seed, mode, method_name):
    """
    取得其他方法 (DANN 等) 的 MDE
    路徑結構: {method_name}/experiments/random_seed_{seed}/*_{mode}/predictions/
    """
    base_path = os.path.join(method_name, "experiments", f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return np.nan, np.nan
        
    # === 修改開始: 針對 DNN 做特殊處理 ===
    if method_name == "DNN":
        # DNN 資料夾結構特殊 (例如 SourceOnly_Epoch1)，且不分 labeled/unlabeled
        # 使用 *SourceOnly* 作為關鍵字搜尋
        search_pattern = os.path.join(base_path, "*SourceOnly*")
    else:
        # 其他方法維持原樣，尋找以 labeled/unlabeled 結尾的資料夾
        search_pattern = os.path.join(base_path, f"*{mode}") 
    # === 修改結束 ===

    found_dirs = glob.glob(search_pattern)
    
    target_dir = None
    if found_dirs:
        target_dir = found_dirs[0]
        
    if not target_dir:
        return np.nan, np.nan
        
    pred_dir = os.path.join(target_dir, "predictions")
    src_path = os.path.join(pred_dir, FILE_SOURCE)
    tgt_path = os.path.join(pred_dir, FILE_TARGET)
    
    mde_s = calculate_mde_from_file(src_path)
    mde_t = calculate_mde_from_file(tgt_path)
    
    return mde_s, mde_t


# --- 4. 繪圖核心函式 ---

def plot_combined_comparison(df, title_seed_part, output_filename):
    """
    畫圖函式：左 Labeled, 右 Unlabeled
    """
    modes = ['labeled', 'unlabeled']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f'MDE Comparison - {title_seed_part}', fontsize=16)
    
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
            
        # 繪圖順序: WD -> SWD -> Others (包含 DNN)
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
        ax.set_xticklabels(plot_names, rotation=45, ha='right')
        
        ax.set_ylim(0, global_max * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend()
            
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> 圖表已儲存: {output_filename}")


# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_seeds_data = [] 

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"正在處理 Random Seed: {seed}")
        print(f"{'='*40}")
        
        current_seed_data = []
        
        for mode in ['labeled', 'unlabeled']:
            
            # --- A. 讀取 WD 變體 ---
            for (alpha, beta, name) in WD_VARIANTS:
                mde_s, mde_t = get_wd_mde(seed, mode, alpha, beta)
                
                if not np.isnan(mde_s) and not np.isnan(mde_t):
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Source', 'mde': mde_s
                    })
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Target', 'mde': mde_t
                    })

            # --- B. 讀取 SWD 變體 ---
            for (alpha, beta, name) in SWD_VARIANTS:
                mde_s, mde_t = get_swd_mde(seed, mode, alpha, beta)
                
                if not np.isnan(mde_s) and not np.isnan(mde_t):
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Source', 'mde': mde_s
                    })
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Target', 'mde': mde_t
                    })

            # --- C. 讀取 其他方法 (含 DNN) ---
            for method in OTHER_METHODS:
                mde_s, mde_t = get_other_mde(seed, mode, method)
                
                if not np.isnan(mde_s) and not np.isnan(mde_t):
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': method, 
                        'type': 'Source', 'mde': mde_s
                    })
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': method, 
                        'type': 'Target', 'mde': mde_t
                    })

        # --- 該 Seed 資料收集完畢，繪圖 ---
        if current_seed_data:
            df_seed = pd.DataFrame(current_seed_data)
            out_name = os.path.join(OUTPUT_DIR, f"comparison_seed_{seed}.png")
            plot_combined_comparison(df_seed, f"Random Seed {seed}", out_name)
            
            all_seeds_data.extend(current_seed_data)
        else:
            print(f"Seed {seed} 沒有抓到任何有效資料。")

    # --- 2. 計算平均並繪製總圖 ---
    print(f"\n{'='*40}")
    print(f"正在計算所有 Seeds 的平均並繪圖...")
    
    if all_seeds_data:
        df_all = pd.DataFrame(all_seeds_data)
        df_avg = df_all.groupby(['mode', 'method_name', 'type'], as_index=False)['mde'].mean()
        
        out_name_avg = os.path.join(OUTPUT_DIR, "comparison_average.png")
        plot_combined_comparison(df_avg, "Average Result", out_name_avg)
    else:
        print("沒有足夠的資料可以計算平均。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()