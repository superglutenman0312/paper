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
    # exit() 
except Exception as e:
    print(f"載入 {LABEL_MAP_PATH} 時發生錯誤: {e}")
    exit()

# --- 2. (保留) 'Evaluator' class 相關函式 ---
def class_to_coordinate(a):
    try:
        return LABEL_TO_COORDINATE_DICT[a]
    except KeyError:
        raise KeyError(f"標籤 {a} 在載入的字典中找不到。")

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

# --- 4. (修改) 繪圖函式：支援自訂標題字串 ---
def plot_combined_chart(df, title_seed_part, experiment_name, output_filename):
    """
    繪製合併長條圖 (左: Labeled, 右: Unlabeled)
    title_seed_part: 標題中關於 Seed 的描述，例如 "Random Seed 42" 或 "Average Result"
    """
    modes = ['labeled', 'unlabeled']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)) 
    
    # 設定整張圖的大標題 (使用傳入的 title_seed_part)
    fig.suptitle(f'MDE Analysis - {title_seed_part}\nExperiment: {experiment_name}', fontsize=16)

    for i, mode in enumerate(modes):
        ax = axes[i]
        
        mode_df = df[df['mode'] == mode].copy()
        if mode_df.empty:
            print(f"  [Skip] {title_seed_part} - {mode}: 沒有資料可供繪圖。")
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{mode.capitalize()} Mode')
            continue

        # 確保按照 Beta 大小排序
        mode_df = mode_df.sort_values(by=['beta'])

        combo_labels = mode_df['combo_label'].unique()
        source_mdes = mode_df[mode_df['type'] == 'Source']['mde'].values
        target_mdes = mode_df[mode_df['type'] == 'Target']['mde'].values

        x = np.arange(len(combo_labels))
        width = 0.35 

        # 繪製長條圖
        rects1 = ax.bar(x - width/2, source_mdes, width, label='Source MDE', alpha=0.9)
        rects2 = ax.bar(x + width/2, target_mdes, width, label='Target MDE', alpha=0.9)

        ax.set_ylabel('Mean Distance Error (MDE) [m]')
        ax.set_xlabel('Loss Weights Parameters')
        ax.set_title(f'{mode.capitalize()} Mode')
        
        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, rotation=0)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"  成功儲存圖表: {output_filename}")


# --- 5. (修改) 主程式：加入平均值計算邏輯 ---
def main():
    
    # --- 參數設定 ---
    EXPERIMENT_TYPE = 'experiments4' 
    MODES = ['labeled', 'unlabeled']
    ALPHAS = [1.0] 
    BETAS = [1.0, 10.0, 100.0]
    EPOCH = 100
    RANDOM_SEEDS = [42, 70, 100] 
    
    if EXPERIMENT_TYPE == 'experiments4':
        SOURCE_FILE_TAG = '211120' 
        TARGET_FILE_TAG = '221221' 
    else:
        print("請檢查 EXPERIMENT_TYPE 設定")
        return
        
    base_work_dir = EXPERIMENT_TYPE
    
    print(f"開始分析 [個別種子 + 平均結果繪圖模式]")
    print(f"實驗路徑: {base_work_dir}")

    # 用來儲存所有 Seed 的結果，以便最後算平均
    all_results = []

    # --- 遍歷每個 Random Seed ---
    for seed in RANDOM_SEEDS:
        print(f"\n{'='*40}")
        print(f"正在處理 Random Seed: {seed}")
        print(f"{'='*40}")
        
        seed_results = []
        seed_base_dir = os.path.join(base_work_dir, f'random_seed_{seed}')
        
        if not os.path.exists(seed_base_dir):
            print(f"警告: 找不到資料夾 {seed_base_dir}，跳過此種子。")
            continue

        # 蒐集資料
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
                        # 建立單筆資料字典
                        record_s = { 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Source', 'mde': mde_source }
                        record_t = { 'mode': mode, 'alpha': alpha, 'beta': beta, 'combo_label': combo_label, 'type': 'Target', 'mde': mde_target }
                        
                        seed_results.append(record_s)
                        seed_results.append(record_t)

        # 將此 Seed 的結果加入總表
        all_results.extend(seed_results)

        if not seed_results:
            print(f"Seed {seed} 沒有收集到任何有效數據。")
            continue
            
        # 繪製個別 Seed 的圖
        seed_df = pd.DataFrame(seed_results)
        combined_out_path = os.path.join(seed_base_dir, f'seed_{seed}_combined_mde.png')
        plot_combined_chart(seed_df, f"Random Seed {seed}", EXPERIMENT_TYPE, combined_out_path)

    # --- 最後：計算平均並繪製總圖 ---
    print(f"\n{'='*40}")
    print(f"正在計算所有 Seeds 的平均並繪圖...")
    print(f"{'='*40}")

    if all_results:
        all_df = pd.DataFrame(all_results)
        
        # 根據 模式、參數、類型 分組取平均
        # 注意: combo_label 是由 alpha/beta 決定的，所以也可以放進 groupby
        avg_df = all_df.groupby(['mode', 'alpha', 'beta', 'combo_label', 'type'], as_index=False)['mde'].mean()
        
        # 設定平均圖的輸出路徑 (在 experiments4/ 底下)
        avg_output_filename = os.path.join(base_work_dir, 'average_combined_mde.png')
        
        # 繪製平均圖
        plot_combined_chart(avg_df, "Average Result", EXPERIMENT_TYPE, avg_output_filename)
        
    else:
        print("錯誤：沒有任何數據可供計算平均值。")

    print("\n所有作業完成！")

if __name__ == "__main__":
    main()