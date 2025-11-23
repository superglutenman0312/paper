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

# 2. 定義要比較的 SWD 參數 (Alpha 固定 1, Beta 變化)
#    格式: (Alpha, Beta, 顯示名稱)
SWD_VARIANTS = [
    (1.0, 1.0, "SWD_1_1"),
    (1.0, 10.0, "SWD_1_10"),
    (1.0, 100.0, "SWD_1_100")
]

# 3. 定義其他方法 (對應資料夾名稱)
OTHER_METHODS = ["DANN", "DANN_CORR"]

# 4. 檔案標籤 (Source / Target)
FILE_SOURCE = "211120_results.csv"
FILE_TARGET = "221221_results.csv"

# 5. Label Map 路徑 (請確保正確)
LABEL_MAP_PATH = 'D:/paper_thesis/My/data/MTLocData/Mall/label_map.pkl'

# 6. 路徑設定 (假設此腳本在 MALL/ 下)
#    SWD 資料夾在 MALL/SWD/experiments4
SWD_BASE_DIR = os.path.join("SWD", "experiments4")
OUTPUT_DIR = "Comparison_Result"

# ================= 工具函式 =================

# --- 1. 載入 Label Map ---
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    # 反轉字典: { (x,y): label } -> { label: (x,y) }
    # 或是如果原始就是 {label: (x,y)}，請根據實際情況調整。
    # 根據您提供的 plot_others_mde.py，原始似乎是 { (x,y): label }，所以這裡做反轉
    LABEL_TO_COORDINATE_DICT = {value: key for key, value in label_map.items()}
    print(f"成功載入座標對照表，共 {len(LABEL_TO_COORDINATE_DICT)} 筆資料。")
except Exception as e:
    print(f"[錯誤] 無法載入 Label Map: {e}")
    # 為了避免程式直接崩潰，這裡定義一個空的或假的，但在實際執行時應該要修好路徑
    LABEL_TO_COORDINATE_DICT = {}

def class_to_coordinate(a):
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        return None # 或拋出錯誤

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
    """
    取得 SWD 特定參數下的 MDE
    路徑結構: SWD/experiments4/random_seed_{seed}/{alpha}_{beta}_{epoch}_{mode}/predictions/
    """
    folder_name = f"{float(alpha)}_{float(beta)}_{EPOCH_SWD}_{mode}"
    model_dir = os.path.join(SWD_BASE_DIR, f"random_seed_{seed}", folder_name, "predictions")
    
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
        
    # 搜尋結尾是 _mode 的資料夾 (例如 _labeled)
    search_pattern = os.path.join(base_path, f"*{mode}") # 注意: 這裡假設資料夾名稱結尾直接是 labeled/unlabeled
    found_dirs = glob.glob(search_pattern)
    
    # 有些資料夾命名可能是 _labeled，有些可能是 _1_labeled，這裡做個簡單過濾
    target_dir = None
    if found_dirs:
        # 優先找完全符合 suffix 的，或者取第一個
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
    df 需要包含欄位: ['mode', 'method_name', 'type', 'mde']
    """
    modes = ['labeled', 'unlabeled']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'MDE Comparison - {title_seed_part}', fontsize=16)
    
    # 定義顏色
    color_source = '#4c72b0' 
    color_target = '#dd8452'
    
    # 找出所有數據的最大值，統一 Y 軸
    global_max = df['mde'].max() if not df.empty else 1.0

    for i, mode in enumerate(modes):
        ax = axes[i]
        
        # 篩選該模式的資料
        mode_df = df[df['mode'] == mode].copy()
        
        if mode_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            ax.set_title(f'{mode.capitalize()} Mode')
            continue
            
        # 確保順序：先 SWD (依照 variants 列表順序)，再 Others
        # 我們利用一個自定義排序列表
        ordered_names = [v[2] for v in SWD_VARIANTS] + OTHER_METHODS
        
        # 建立繪圖用的列表，確保順序正確
        plot_names = []
        source_vals = []
        target_vals = []
        
        for name in ordered_names:
            # 檢查這個方法在這個模式下有沒有資料
            subset = mode_df[mode_df['method_name'] == name]
            if not subset.empty:
                # 取出 Source 和 Target 的值
                val_s = subset[subset['type'] == 'Source']['mde'].values
                val_t = subset[subset['type'] == 'Target']['mde'].values
                
                # 如果有值 (非 NaN)，則加入繪圖清單
                # 注意：這裡假設每個方法只有一筆資料 (因為是單一 Seed 或 平均值)
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
        ax.set_xticklabels(plot_names, rotation=30, ha='right')
        
        ax.set_ylim(0, global_max * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend()
            
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=9)
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  >> 圖表已儲存: {output_filename}")


# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_seeds_data = [] # 用來存所有 Seed 的資料以算平均

    # --- 1. 遍歷每個 Random Seed ---
    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"正在處理 Random Seed: {seed}")
        print(f"{'='*40}")
        
        current_seed_data = []
        
        for mode in ['labeled', 'unlabeled']:
            
            # --- A. 讀取 SWD 變體 ---
            for (alpha, beta, name) in SWD_VARIANTS:
                mde_s, mde_t = get_swd_mde(seed, mode, alpha, beta)
                
                if not np.isnan(mde_s) and not np.isnan(mde_t):
                    # 存入列表 (Source)
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Source', 'mde': mde_s
                    })
                    # 存入列表 (Target)
                    current_seed_data.append({
                        'seed': seed, 'mode': mode, 'method_name': name, 
                        'type': 'Target', 'mde': mde_t
                    })
                else:
                    # 雖然是 NaN 也可以選擇是否要印出 debug
                    pass

            # --- B. 讀取 其他方法 ---
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
            
            # 加入總表
            all_seeds_data.extend(current_seed_data)
        else:
            print(f"Seed {seed} 沒有抓到任何有效資料。")

    # --- 2. 計算平均並繪製總圖 ---
    print(f"\n{'='*40}")
    print(f"正在計算所有 Seeds 的平均並繪圖...")
    
    if all_seeds_data:
        df_all = pd.DataFrame(all_seeds_data)
        
        # 根據 mode, method_name, type 進行分組平均
        df_avg = df_all.groupby(['mode', 'method_name', 'type'], as_index=False)['mde'].mean()
        
        out_name_avg = os.path.join(OUTPUT_DIR, "comparison_average.png")
        plot_combined_comparison(df_avg, "Average Result", out_name_avg)
    else:
        print("沒有足夠的資料可以計算平均。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()