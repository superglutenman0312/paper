import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 設定區 =================

# 1. 定義要比較的方法 (對應資料夾名稱)
METHODS = ["DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 2. 定義 Random Seeds
SEEDS = [42, 70, 100]

# 3. 定義場景與對應的 Source/Target 檔案
# Time Reversal 邏輯: Source 都是 2020-02-19
SCENARIOS = {
    "time_reversal_1": {
        "source_file": "200219_results.csv",  # Source Domain
        "target_file": "190611_results.csv"   # Target Domain
    },
    "time_reversal_2": {
        "source_file": "200219_results.csv",  # Source Domain
        "target_file": "191009_results.csv"   # Target Domain
    }
}

# 4. 輸出目錄
OUTPUT_DIR = "MDE_result_UM_DSI_reverse"

# 5. 內建座標對照表 (來自 evaluator.py)
LABEL_TO_COORD = {
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

# ================= 工具函式 =================

def calculate_mde_from_labels(csv_path):
    """
    讀取 CSV，利用 LABEL_TO_COORD 將 label/pred 轉為座標，計算 MDE
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns or 'pred' not in df.columns:
            print(f"  [警告] 欄位缺失 ({os.path.basename(csv_path)})")
            return None

        # Map labels to coordinates
        df['true_coord'] = df['label'].map(LABEL_TO_COORD)
        df['pred_coord'] = df['pred'].map(LABEL_TO_COORD)

        # Drop NaNs (invalid labels)
        if df['true_coord'].isnull().any() or df['pred_coord'].isnull().any():
            df = df.dropna(subset=['true_coord', 'pred_coord'])

        if len(df) == 0:
            return None

        # Stack into numpy arrays
        true_xy = np.vstack(df['true_coord'].values)
        pred_xy = np.vstack(df['pred_coord'].values)

        # Calculate Euclidean distance
        distances = np.linalg.norm(true_xy - pred_xy, axis=1)
        
        return distances.mean()

    except Exception as e:
        print(f"  [錯誤] 計算失敗 ({os.path.basename(csv_path)}): {e}")
        return None

def get_prediction_dir(method, scenario, seed, mode_suffix):
    """
    尋找路徑: Method/Scenario/Seed/*mode_suffix/predictions
    """
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

    # 遍歷場景
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
                        s_mde = calculate_mde_from_labels(os.path.join(pred_dir, src_file))
                        # 算 Target MDE
                        t_mde = calculate_mde_from_labels(os.path.join(pred_dir, tgt_file))

                        if s_mde is not None and t_mde is not None:
                            plot_data[mode][method] = (s_mde, t_mde)
                            print(f"    [{method}-{mode}] Src: {s_mde:.2f}, Tgt: {t_mde:.2f}")
                        else:
                            print(f"    [{method}-{mode}] 檔案缺失")
                    else:
                        # print(f"    [{method}-{mode}] 找不到資料夾")
                        pass

            # --- 繪圖 ---
            if not plot_data['Labeled'] and not plot_data['Unlabeled']:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'MDE - {scen_name} (Seed {seed})', fontsize=16)

            color_source = '#4c72b0'
            color_target = '#dd8452'
            bar_width = 0.35

            # 統一 Y 軸
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
                for rects in [rects1, rects2]:
                    for rect in rects:
                        h = rect.get_height()
                        ax.annotate(f'{h:.2f}',
                                    xy=(rect.get_x() + rect.get_width()/2, h),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(OUTPUT_DIR, f"{scen_name}_seed_{seed}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  >> 圖表已儲存: {save_path}")

if __name__ == "__main__":
    main()