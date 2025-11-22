import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 核心函式 (座標轉換與距離計算) ---
# (針對 MCSL 資料集的 12x12 Grid 設定)

def class_to_coordinate(a):
    """
    使用 NumPy 2D 陣列 (table) 將標籤 ID 轉換為 (x, y) 座標。
    """
    table = np.array([
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
    
    locations = np.argwhere(table == a)
    if locations.size == 0:
        # 某些標籤可能不在這張表內 (例如 0 或其他)
        # 這裡可以選擇 raise error 或 return None
        raise KeyError(f"標籤 {a} 在 table 中找不到。")
        
    x = locations[0][1]  # 欄索引 (column index)
    y = locations[0][0]  # 列索引 (row index)
    coordinate = [x, y]
    return coordinate

def euclidean_distance(p1, p2):
    """
    計算歐幾里得距離，並乘以 0.6 的縮放因子 (Grid Size)。
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) * 0.6

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


# --- 2. 繪圖函式 ---

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
    # 這次我們一次跑兩個實驗類型
    EXPERIMENT_TYPES = ['time_variation2', 'spatial_variation2']
    
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [1.0] # Alpha 固定為 1
    BETAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    EPOCH = 500
    RANDOM_SEEDS = [42, 70, 100] 
    # BETAS = [0.1, 1.0]
    # EPOCH = 1
    # RANDOM_SEEDS = [42, 70]

    # --- 最外層：遍歷實驗類型 (Time vs Spatial) ---
    for exp_type in EXPERIMENT_TYPES:
        print(f"\n{'#'*60}")
        print(f"開始分析實驗類型: {exp_type}")
        print(f"{'#'*60}")

        # 設定檔案標籤
        if exp_type == 'time_variation2':
            SOURCE_FILE_TAG = '220318'
            TARGET_FILE_TAG = '231116'
        elif exp_type == 'spatial_variation2':
            SOURCE_FILE_TAG = '231116'
            TARGET_FILE_TAG = '231117'
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