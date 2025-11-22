import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# --- 1. (保留) 載入並反轉座標/標籤對照表 ---
LABEL_MAP_PATH = 'D:/paper_thesis/My/data/MTLocData/Mall/label_map.pkl' 

try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f) 
    LABEL_TO_COORDINATE_DICT = {value: key for key, value in label_map.items()} 
    print(f"成功載入並反轉座標對照表: {LABEL_MAP_PATH}")
except FileNotFoundError:
    print(f"錯誤：找不到 'label_map.pkl' 檔案於: {LABEL_MAP_PATH}")
    exit()
except Exception as e:
    print(f"載入 {LABEL_MAP_PATH} 時發生錯誤: {e}")
    exit()

# --- 2. (保留) 'Evaluator' class 相關函式 ---
def class_to_coordinate(a):
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        raise KeyError(f"標籤 {a} 在 {LABEL_MAP_PATH} 載入的字典中找不到。")

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- 3. (保留) 計算 MDE 的主函式 ---
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
            distance_error = euclidean_distance(pred_coord, actual_coord)
            errors.append(distance_error)
        except Exception:
            continue

    if errors:
        return np.mean(errors)
    else:
        return np.nan

# --- 4. (還原並修改) 繪圖函式：繪製單一種子的結果 ---
def plot_single_seed_chart(df, mode, seed, experiment_name, output_filename):
    """
    繪製單一 Random Seed 的長條圖 (無誤差線)
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
    
    # 繪製單純的長條圖
    rects1 = ax.bar(x - width/2, source_mdes, width, label='Source MDE', alpha=0.9)
    rects2 = ax.bar(x + width/2, target_mdes, width, label='Target MDE', alpha=0.9)

    ax.set_ylabel('Mean Distance Error (MDE) [m]')
    ax.set_xlabel('Loss Weights Parameters')
    # 標題顯示這是哪個 Seed
    ax.set_title(f'MDE Analysis ({mode.capitalize()}) - Random Seed {seed}\nExperiment: {experiment_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  成功儲存圖表: {output_filename}")


# --- 5. (修改) 主程式：針對每個 Seed 獨立繪圖 ---
def main():
    
    # --- 參數設定 ---
    EXPERIMENT_TYPE = 'experiments4' 
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [1.0] 
    BETAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    EPOCH = 100
    RANDOM_SEEDS = [42, 70, 100] 
    # BETAS = [0.1, 1.0]
    # EPOCH = 1
    # RANDOM_SEEDS = [42, 70]
    
    if EXPERIMENT_TYPE == 'experiments4':
        SOURCE_FILE_TAG = '211120' 
        TARGET_FILE_TAG = '221221' 
    else:
        print("請檢查 EXPERIMENT_TYPE 設定")
        return
        
    base_work_dir = EXPERIMENT_TYPE
    
    print(f"開始分析 [個別種子繪圖模式]")
    print(f"實驗路徑: {base_work_dir}")

    # --- 最外層迴圈：遍歷每個 Random Seed ---
    for seed in RANDOM_SEEDS:
        print(f"\n{'='*40}")
        print(f"正在處理 Random Seed: {seed}")
        print(f"{'='*40}")
        
        # !! 關鍵 !! 每個 Seed 都有自己獨立的結果列表
        seed_results = []
        
        # 設定該 Seed 的根目錄，圖表將存在這裡
        seed_base_dir = os.path.join(base_work_dir, f'random_seed_{seed}')
        
        if not os.path.exists(seed_base_dir):
            print(f"警告: 找不到資料夾 {seed_base_dir}，跳過此種子。")
            continue

        # 蒐集該 Seed 下的所有參數組合數據
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
            print(f"Seed {seed} 沒有收集到任何有效數據。")
            continue
            
        seed_df = pd.DataFrame(seed_results)
        
        # --- 繪圖並存檔 (存在該 Seed 的資料夾內) ---
        # 1. Labeled
        labeled_out_path = os.path.join(seed_base_dir, f'seed_{seed}_labeled_mde.png')
        plot_single_seed_chart(seed_df, 'labeled', seed, EXPERIMENT_TYPE, labeled_out_path)
        
        # 2. Unlabeled
        unlabeled_out_path = os.path.join(seed_base_dir, f'seed_{seed}_unlabeled_mde.png')
        plot_single_seed_chart(seed_df, 'unlabeled', seed, EXPERIMENT_TYPE, unlabeled_out_path)

    print("\n所有個別種子的圖表繪製完成！")

if __name__ == "__main__":
    main()