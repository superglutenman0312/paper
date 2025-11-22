import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 設定區 =================

# 1. 定義要比較的方法
METHODS = ["DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 2. 定義 Random Seeds
# SEEDS = [42, 70, 100]
SEEDS = [42, 70]

# 3. 定義場景與對應的 Source/Target 檔案
# 根據 Train Script 與目錄截圖設定
SCENARIOS = {
    "time_variation": {
        "source_file": "220318_results.csv",
        "target_file": "231116_results.csv"
    },
    "spatial_variation": {
        "source_file": "231116_results.csv",
        "target_file": "231117_results.csv"
    }
}

# 4. 輸出目錄
OUTPUT_DIR = "MDE_result_MCSL"

# ================= 核心算法 (源自 evaluator.py) =================

# 定義格點 Table (12x12)
MCSL_TABLE = np.array([
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

def build_label_map(table):
    """
    將 2D table 預處理為 {label: (x, y)} 的字典，加快查詢速度。
    x = column index, y = row index
    """
    label_map = {}
    rows, cols = table.shape
    for r in range(rows):
        for c in range(cols):
            label = table[r, c]
            if label != 0: # 假設 0 是空位或背景，不處理
                label_map[label] = (c, r) # 注意: evaluator.py 定義 coordinate = [x, y]
    return label_map

# 初始化全域對照表
LABEL_TO_COORD = build_label_map(MCSL_TABLE)

def calculate_mde_mcsl(csv_path):
    """
    MCSL 專用 MDE 計算：
    1. 查表取得座標
    2. 計算歐式距離
    3. 乘以 0.6 縮放因子
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns or 'pred' not in df.columns:
            print(f"  [警告] 欄位缺失 ({os.path.basename(csv_path)})")
            return None

        # 確保是整數
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        df['pred'] = pd.to_numeric(df['pred'], errors='coerce').fillna(0).astype(int)

        # 映射座標
        df['true_coord'] = df['label'].map(LABEL_TO_COORD)
        df['pred_coord'] = df['pred'].map(LABEL_TO_COORD)

        # 移除查不到座標的資料 (例如 label=0 或 table 中不存在的 label)
        if df['true_coord'].isnull().any() or df['pred_coord'].isnull().any():
            df = df.dropna(subset=['true_coord', 'pred_coord'])

        if len(df) == 0:
            return None

        # 轉為 NumPy Array 進行計算
        true_xy = np.vstack(df['true_coord'].values)
        pred_xy = np.vstack(df['pred_coord'].values)

        # 歐式距離計算
        raw_distances = np.linalg.norm(true_xy - pred_xy, axis=1)
        
        # 乘以縮放因子 0.6 (依據 evaluator.py)
        final_distances = raw_distances * 0.6
        
        return final_distances.mean()

    except Exception as e:
        print(f"  [錯誤] 計算失敗 ({os.path.basename(csv_path)}): {e}")
        return None

# ================= 工具函式 =================

def get_prediction_dir(method, scenario, seed, mode_suffix):
    """
    尋找路徑: MCSL/Method/Scenario/Seed/*mode_suffix/predictions
    """
    # 建構基礎路徑
    base_path = os.path.join(method, scenario, f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return None

    # 尋找 *_labeled 或 *_unlabeled 資料夾
    search_pattern = os.path.join(base_path, f"*{mode_suffix}")
    found_dirs = glob.glob(search_pattern)

    if not found_dirs:
        return None
    
    # 進入 predictions
    pred_dir = os.path.join(found_dirs[0], "predictions")
    if os.path.exists(pred_dir):
        return pred_dir
    return None

# ================= 主程式 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"建立輸出目錄: {OUTPUT_DIR}")
        print(f"已載入 MCSL Table，共 {len(LABEL_TO_COORD)} 個有效座標點。")

    # 遍歷場景 (Time / Spatial)
    for scen_name, scen_files in SCENARIOS.items():
        print(f"\n正在處理場景: {scen_name} ...")
        
        src_file = scen_files['source_file']
        tgt_file = scen_files['target_file']

        # 遍歷 Seeds
        for seed in SEEDS:
            print(f"  -> Seed: {seed}")
            
            plot_data = {'Labeled': {}, 'Unlabeled': {}}
            
            for method in METHODS:
                for mode in ['Labeled', 'Unlabeled']:
                    suffix = "_labeled" if mode == 'Labeled' else "_unlabeled"
                    
                    pred_dir = get_prediction_dir(method, scen_name, seed, suffix)
                    
                    if pred_dir:
                        # 算 Source MDE
                        s_mde = calculate_mde_mcsl(os.path.join(pred_dir, src_file))
                        # 算 Target MDE
                        t_mde = calculate_mde_mcsl(os.path.join(pred_dir, tgt_file))

                        if s_mde is not None and t_mde is not None:
                            plot_data[mode][method] = (s_mde, t_mde)
                            print(f"    [{method}-{mode}] Src: {s_mde:.2f}, Tgt: {t_mde:.2f}")
                        else:
                            print(f"    [{method}-{mode}] 檔案缺失")
                    else:
                        pass # 找不到資料夾保持安靜

            # --- 繪圖 ---
            if not plot_data['Labeled'] and not plot_data['Unlabeled']:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'MDE (MCSL) - {scen_name} (Seed {seed})', fontsize=16)

            color_source = '#4c72b0'
            color_target = '#dd8452'
            bar_width = 0.35

            # 計算 Y 軸最大值
            all_vals = []
            for d in [plot_data['Labeled'], plot_data['Unlabeled']]:
                for v in d.values():
                    all_vals.extend(v)
            global_max = max(all_vals) if all_vals else 1.0

            for idx, mode in enumerate(['Labeled', 'Unlabeled']):
                ax = axes[idx]
                data = plot_data[mode]

                if not data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
                    ax.set_title(f'{mode} Training', fontsize=14)
                    continue

                methods_present = list(data.keys())
                s_vals = [data[m][0] for m in methods_present]
                t_vals = [data[m][1] for m in methods_present]
                x = np.arange(len(methods_present))

                rects1 = ax.bar(x - bar_width/2, s_vals, bar_width, label='Source', color=color_source, alpha=0.8)
                rects2 = ax.bar(x + bar_width/2, t_vals, bar_width, label='Target', color=color_target, alpha=0.8)

                ax.set_title(f'{mode} Training', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(methods_present, fontsize=11)
                ax.set_ylabel('MDE (m)')
                ax.set_ylim(0, global_max * 1.2)
                ax.grid(axis='y', linestyle='--', alpha=0.5)

                if idx == 0:
                    ax.legend()

                # 標籤
                def autolabel(rects):
                    for rect in rects:
                        h = rect.get_height()
                        ax.annotate(f'{h:.2f}',
                                    xy=(rect.get_x() + rect.get_width()/2, h),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                autolabel(rects1)
                autolabel(rects2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(OUTPUT_DIR, f"{scen_name}_seed_{seed}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  >> 圖表已儲存: {save_path}")

if __name__ == "__main__":
    main()