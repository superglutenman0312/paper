import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ================= 設定區 =================

# 1. 定義要比較的方法 (對應資料夾名稱)
# METHODS = ["DANN", "DANN_CORR", "DANN_CORR_GEMINI"]
METHODS = ["DANN", "DANN_GEMINI"]

# 2. 定義 Random Seeds
SEEDS = [42, 70, 100]

# 3. 定義場景與對應的 Target 檔案
# 根據截圖: 190611 是 Source
SCENARIOS = {
    "time_variation_1": {
        "source_file": "190611_results.csv",
        "target_file": "191009_results.csv" 
    },
    "time_variation_2": {
        "source_file": "190611_results.csv",
        "target_file": "200219_results.csv"
    }
}

# 4. 輸出目錄
OUTPUT_DIR = "MDE_result_UM_DSI2"

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
    讀取包含 label, pred 的 CSV，轉換成座標後計算 MDE
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        # 檢查必要欄位
        if 'label' not in df.columns or 'pred' not in df.columns:
            print(f"  [警告] 欄位錯誤 ({os.path.basename(csv_path)}): 找不到 'label' 或 'pred'")
            return None

        # 利用 map 將 label 轉換為座標 (tuple)
        df['true_coord'] = df['label'].map(LABEL_TO_COORD)
        df['pred_coord'] = df['pred'].map(LABEL_TO_COORD)

        # 移除無效轉換
        if df['true_coord'].isnull().any() or df['pred_coord'].isnull().any():
            df = df.dropna(subset=['true_coord', 'pred_coord'])

        if len(df) == 0:
            return None

        # 將 tuple 拆解成 numpy array
        true_xy = np.vstack(df['true_coord'].values)
        pred_xy = np.vstack(df['pred_coord'].values)

        # 計算歐式距離
        distances = np.linalg.norm(true_xy - pred_xy, axis=1)
        
        return distances.mean()

    except Exception as e:
        print(f"  [錯誤] 計算 MDE 失敗 ({os.path.basename(csv_path)}): {e}")
        return None

def get_prediction_dir(method, scenario, seed, mode_suffix):
    """
    尋找 predictions 資料夾路徑
    路徑範例: DANN/time_variation_1/random_seed_42/1.0_0.0_1_labeled/predictions
    """
    # 1. 定位到 Random Seed 層級
    base_path = os.path.join(method, scenario, f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return None

    # 2. 搜尋包含 mode_suffix 的資料夾 (e.g., *_labeled)
    # 注意：這裡假設只有一個符合的資料夾
    search_pattern = os.path.join(base_path, f"*{mode_suffix}")
    found_dirs = glob.glob(search_pattern)

    if not found_dirs:
        return None
    
    # 3. 進入 predictions 子資料夾
    pred_dir = os.path.join(found_dirs[0], "predictions")
    
    if os.path.exists(pred_dir):
        return pred_dir
    else:
        return None

# ================= 主程式邏輯 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"建立輸出目錄: {OUTPUT_DIR}")

    # 針對每個場景分別畫圖 (Time Variation 1 & 2)
    for scen_name, scen_files in SCENARIOS.items():
        print(f"\n正在處理場景: {scen_name} ...")
        
        source_filename = scen_files['source_file']
        target_filename = scen_files['target_file']

        # 針對每個 Random Seed 畫一張圖
        for seed in SEEDS:
            print(f"  -> Seed: {seed}")
            
            plot_data = {'Labeled': {}, 'Unlabeled': {}}
            
            for method in METHODS:
                for mode in ['Labeled', 'Unlabeled']:
                    suffix = "_labeled" if mode == 'Labeled' else "_unlabeled"
                    
                    # 取得預測資料夾路徑
                    pred_dir = get_prediction_dir(method, scen_name, seed, suffix)
                    
                    if pred_dir:
                        # 計算 Source MDE
                        src_path = os.path.join(pred_dir, source_filename)
                        source_mde = calculate_mde_from_labels(src_path)
                        
                        # 計算 Target MDE
                        tgt_path = os.path.join(pred_dir, target_filename)
                        target_mde = calculate_mde_from_labels(tgt_path)

                        if source_mde is not None and target_mde is not None:
                            plot_data[mode][method] = (source_mde, target_mde)
                            print(f"    [{method}-{mode}] Src: {source_mde:.2f}, Tgt: {target_mde:.2f}")
                        else:
                            print(f"    [{method}-{mode}] 缺檔")
                    else:
                        # print(f"    [{method}-{mode}] 資料夾找不到")
                        pass

            # --- 繪圖階段 ---
            if not plot_data['Labeled'] and not plot_data['Unlabeled']:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            # 標題包含場景和種子
            fig.suptitle(f'MDE - {scen_name} (Seed {seed})', fontsize=16)

            color_source = '#4c72b0'
            color_target = '#dd8452'
            bar_width = 0.35

            # 統一 Y 軸範圍
            all_values = []
            for m_data in [plot_data['Labeled'], plot_data['Unlabeled']]:
                for v in m_data.values():
                    all_values.extend(v)
            
            # 如果沒有數據，預設 y_max = 1
            global_max = max(all_values) if all_values else 1.0

            modes_list = ['Labeled', 'Unlabeled']
            
            for idx, mode in enumerate(modes_list):
                ax = axes[idx]
                data = plot_data[mode]
                
                if not data:
                    ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
                    ax.set_title(f'{mode} Training', fontsize=14)
                    continue

                methods_present = list(data.keys())
                source_vals = [data[m][0] for m in methods_present]
                target_vals = [data[m][1] for m in methods_present]
                
                x = np.arange(len(methods_present))

                rects1 = ax.bar(x - bar_width/2, source_vals, bar_width, label='Source Domain', color=color_source, alpha=0.8)
                rects2 = ax.bar(x + bar_width/2, target_vals, bar_width, label='Target Domain', color=color_target, alpha=0.8)

                ax.set_title(f'{mode} Training', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(methods_present, fontsize=11)
                ax.set_ylabel('MDE (meters)')
                ax.set_ylim(0, global_max * 1.2)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                
                if idx == 0:
                    ax.legend()

                # 標示數值
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

                autolabel(rects1)
                autolabel(rects2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 檔名包含場景名稱
            save_filename = f"{scen_name}_seed_{seed}.png"
            save_path = os.path.join(OUTPUT_DIR, save_filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  >> 圖表已儲存: {save_path}")

if __name__ == "__main__":
    main()