import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ================= 設定區 (已根據圖片修正) =================

# 1. 定義要比較的方法 (對應資料夾名稱)
METHODS = ["DANN", "DANN_CORR", "DANN_CORR_GEMINI"]

# 2. 定義 Random Seeds
SEEDS = [42, 70, 100]

# 3. 定義檔案名稱 (Source 和 Target)
# 修正：根據圖片，兩個檔案都有 's' (results)
FILE_SOURCE = "211120_results.csv"   
FILE_TARGET = "221221_results.csv"

# 4. Label Map 的路徑 (請確認此路徑是否正確)
LABEL_MAP_PATH = 'D:/paper_thesis/My/data/MTLocData/Mall/label_map.pkl'

# 5. 輸出目錄
OUTPUT_DIR = "MDE_result"

# ================= 初始化與工具函式 =================

def load_label_map(pkl_path):
    """
    載入 label_map.pkl 並建立 {label: (x, y)} 的反向對照表
    """
    if not os.path.exists(pkl_path):
        print(f"[嚴重錯誤] 找不到 Label Map 檔案: {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            label_map = pickle.load(f)
        
        # 反轉字典: { (x,y): label } -> { label: (x,y) }
        label_to_coord = {value: key for key, value in label_map.items()}
        print(f"已成功載入 Label Map，共 {len(label_to_coord)} 個標籤。")
        return label_to_coord
    except Exception as e:
        print(f"[嚴重錯誤] 讀取 Label Map 失敗: {e}")
        return None

def calculate_mde_from_labels(csv_path, label_to_coord):
    """
    讀取包含 label, pred 的 CSV，轉換成座標後計算 MDE
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        # 檢查必要欄位
        if 'label' not in df.columns or 'pred' not in df.columns:
            print(f"  [跳過] 欄位錯誤 ({os.path.basename(csv_path)}): 找不到 'label' 或 'pred'")
            return None

        # 利用 map 將 label 轉換為座標 (tuple)
        df['true_coord'] = df['label'].map(label_to_coord)
        df['pred_coord'] = df['pred'].map(label_to_coord)

        # 檢查是否有無效的轉換 (NaN)並移除
        if df['true_coord'].isnull().any() or df['pred_coord'].isnull().any():
            df = df.dropna(subset=['true_coord', 'pred_coord'])

        if len(df) == 0:
            return None

        # 將 tuple 拆解成 numpy array 進行向量化計算
        true_xy = np.vstack(df['true_coord'].values)
        pred_xy = np.vstack(df['pred_coord'].values)

        # 計算歐式距離
        distances = np.linalg.norm(true_xy - pred_xy, axis=1)
        
        return distances.mean()

    except Exception as e:
        print(f"  [錯誤] 計算 MDE 失敗 ({os.path.basename(csv_path)}): {e}")
        return None

def get_data_path(method, seed, mode_suffix):
    """
    根據方法、種子和模式(_labeled/_unlabeled) 尋找路徑
    """
    # 修正：圖片顯示資料夾名為 random_seed_42 (有底線)
    base_path = os.path.join(method, "experiments", f"random_seed_{seed}")
    
    if not os.path.exists(base_path):
        return None

    # 搜尋結尾是 mode_suffix 的資料夾
    # 例如: 1.0_0.0_1_labeled 會被 *_labeled 抓到
    search_pattern = os.path.join(base_path, f"*{mode_suffix}")
    found_dirs = glob.glob(search_pattern)

    if not found_dirs:
        return None
    
    return found_dirs[0] # 取第一個符合的

# ================= 主程式邏輯 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    label_to_coord = load_label_map(LABEL_MAP_PATH)
    if label_to_coord is None:
        return 

    for seed in SEEDS:
        print(f"\n正在處理 Random Seed: {seed} ...")
        
        plot_data = {'Labeled': {}, 'Unlabeled': {}}
        
        for method in METHODS:
            for mode in ['Labeled', 'Unlabeled']:
                suffix = "_labeled" if mode == 'Labeled' else "_unlabeled"
                
                dir_path = get_data_path(method, seed, suffix)
                
                if dir_path:
                    pred_dir = os.path.join(dir_path, "predictions")
                    
                    # 計算 Source MDE
                    src_path = os.path.join(pred_dir, FILE_SOURCE)
                    source_mde = calculate_mde_from_labels(src_path, label_to_coord)
                    
                    # 計算 Target MDE
                    tgt_path = os.path.join(pred_dir, FILE_TARGET)
                    target_mde = calculate_mde_from_labels(tgt_path, label_to_coord)

                    if source_mde is not None and target_mde is not None:
                        plot_data[mode][method] = (source_mde, target_mde)
                        print(f"  [{method} - {mode}] Source: {source_mde:.4f}, Target: {target_mde:.4f}")
                    else:
                        print(f"  [{method} - {mode}] 數據缺失 (找不到檔案)")
                else:
                    print(f"  [{method} - {mode}] 資料夾不存在 (檢查路徑: {method}/experiments/random_seed_{seed}/*{suffix})")

        # ================= 繪圖階段 =================
        if not plot_data['Labeled'] and not plot_data['Unlabeled']:
            print(f"  [跳過] Seed {seed} 沒有足夠數據繪圖。")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'MDE Comparison - Random Seed {seed}', fontsize=16)

        color_source = '#4c72b0' 
        color_target = '#dd8452'
        bar_width = 0.35
        
        # 設定 Y 軸最大值，讓兩個子圖刻度一致，方便比較
        all_values = []
        for m_data in [plot_data['Labeled'], plot_data['Unlabeled']]:
            for v in m_data.values():
                all_values.extend(v)
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
            
            # 設定統一的 Y 軸範圍
            ax.set_ylim(0, global_max * 1.2) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            if idx == 0:
                ax.legend()

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
        
        save_path = os.path.join(OUTPUT_DIR, f"random_seed_{seed}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  >> 圖表已儲存: {save_path}")

if __name__ == "__main__":
    main()