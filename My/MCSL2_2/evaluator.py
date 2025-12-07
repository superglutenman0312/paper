import pandas as pd
import numpy as np
import argparse
import os
import sys

# ================= 設定區 =================
# 預設的 Map 路徑 (對應到 BLE 資料集的前處理結果)
DEFAULT_MAP_PATH = r"D:\paper_thesis\My\data\MCSL\processed_data\20220318_20231116\experiment_class_map.csv"
# =========================================

def load_coordinate_map(filepath):
    """
    讀取 experiment_class_map.csv 並轉換為字典
    格式: class_id, x, y
    """
    if not os.path.exists(filepath):
        print(f"[Error] 找不到 Map 檔案: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        mapping = {}
        # 假設 csv 欄位是 class_id, x, y
        for _, row in df.iterrows():
            cid = int(row['class_id'])
            x = float(row['x'])
            y = float(row['y'])
            mapping[cid] = (x, y)
        print(f"成功載入 Map: {filepath} (共 {len(mapping)} 個地點)")
        return mapping
    except Exception as e:
        print(f"[Error] 讀取 Map 失敗: {e}")
        return None

def calculate_mde(pred_file, label_map):
    """
    計算 MDE (Mean Distance Error)
    *** 重點：計算出的距離會乘以 0.6 ***
    """
    if not os.path.exists(pred_file):
        print(f"錯誤: 找不到預測結果檔案: {pred_file}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(pred_file)
        # 檢查欄位
        if 'label' not in df.columns or 'pred' not in df.columns:
            print("錯誤: 預測檔案必須包含 'label' 和 'pred' 兩個欄位。")
            sys.exit(1)
            
        errors = []
        valid_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 轉為數值並填補 NaN
                true_id = int(float(row['label']))
                pred_id = int(float(row['pred']))
                
                # 若 ID 為 0 (Masking 的填充值) 通常視為無效或跳過
                if true_id == 0 or pred_id == 0:
                    continue

                if true_id not in label_map or pred_id not in label_map:
                    continue

                true_coord = np.array(label_map[true_id])
                pred_coord = np.array(label_map[pred_id])
                
                # 1. 計算 Grid 歐式距離
                grid_dist = np.linalg.norm(true_coord - pred_coord)
                
                # 2. 【關鍵修改】乘以 0.6 (Grid Size)
                real_dist = grid_dist * 0.6
                
                errors.append(real_dist)
                valid_count += 1
                
            except (ValueError, KeyError):
                continue

        if valid_count == 0:
            print("警告: 沒有計算到任何有效誤差。")
            return None

        errors = np.array(errors)
        mde = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)
        p95 = np.percentile(errors, 95)
        
        return {
            'mde': mde,
            'std': std,
            'median': median,
            'p95': p95,
            'count': valid_count
        }

    except Exception as e:
        print(f"計算過程發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDE for BLE Localization (Grid=0.6m)')
    
    # 預測檔案 (必填)
    parser.add_argument('--file', type=str, required=True, help='預測結果 CSV 路徑')
    
    # Map 檔案 (選填，預設為 processed_data 下的檔案)
    parser.add_argument('--map_file', type=str, default=DEFAULT_MAP_PATH, help='類別座標對照表路徑')
    
    args = parser.parse_args()

    # 1. 載入 Map
    class_map = load_coordinate_map(args.map_file)
    
    if class_map:
        # 2. 計算誤差
        print(f"正在評估: {args.file} ...")
        result = calculate_mde(args.file, class_map)
        
        if result:
            print("\n" + "="*30)
            print(f"評估結果 (樣本數: {result['count']})")
            print("-" * 30)
            print(f"MDE (平均誤差):  {result['mde']:.4f} m (已乘 0.6)")
            print(f"Std (標準差):    {result['std']:.4f} m")
            print(f"Median (中位數): {result['median']:.4f} m")
            print(f"95th Percentile: {result['p95']:.4f} m")
            print("="*30 + "\n")