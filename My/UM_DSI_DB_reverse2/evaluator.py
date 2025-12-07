import pandas as pd
import numpy as np
import argparse
import os
import sys

def load_class_map(map_file):
    """
    讀取類別對照表 (experiment_class_map.csv)
    格式: class_id, x, y
    回傳: {class_id: (x, y)} 字典
    """
    if not os.path.exists(map_file):
        print(f"錯誤: 找不到 Map 檔案: {map_file}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(map_file)
        # 確保 class_id 是整數 (去除可能的浮點數格式)
        label_map = {}
        for _, row in df.iterrows():
            cid = int(row['class_id'])
            # cid = int(row['label_id'])
            x = float(row['x'])
            y = float(row['y'])
            label_map[cid] = (x, y)
            
        print(f"成功載入 Map，共 {len(label_map)} 個地點。")
        return label_map
    except Exception as e:
        print(f"讀取 Map 檔案失敗: {e}")
        sys.exit(1)

def calculate_mde(pred_file, label_map):
    """
    計算 MDE (Mean Distance Error)
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
        missing_count = 0
        
        for idx, row in df.iterrows():
            try:
                true_id = int(row['label'])
                pred_id = int(row['pred'])
                
                # 查表取得座標
                if true_id not in label_map:
                    # 這種情況理論上不該發生 (除非測試資料有 Source 沒去過的地方且沒被過濾)
                    missing_count += 1
                    continue
                    
                if pred_id not in label_map:
                    # 這種情況也不該發生 (除非模型預測出了不存在的 ID)
                    missing_count += 1
                    continue

                true_coord = np.array(label_map[true_id])
                pred_coord = np.array(label_map[pred_id])
                
                # 計算歐式距離 (Euclidean Distance)
                dist = np.linalg.norm(true_coord - pred_coord)
                errors.append(dist)
                valid_count += 1
                
            except ValueError:
                continue

        if valid_count == 0:
            print("警告: 沒有計算到任何有效誤差 (可能是 ID 對不起來)。")
            return None

        errors = np.array(errors)
        mde = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)
        p95 = np.percentile(errors, 95)
        
        if missing_count > 0:
            print(f"警告: 有 {missing_count} 筆資料因 ID 不在 Map 中而被跳過。")

        return {
            'mde': mde,
            'std': std,
            'median': median,
            'p95': p95,
            'count': valid_count,
            'errors': errors # 如果你想畫 CDF，可以用這個
        }

    except Exception as e:
        print(f"計算過程發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDE for Indoor Localization')
    # 注意：這裡建議用 --pred_file 比較明確，且對應下方的 args.pred_file
    parser.add_argument('--file', type=str, required=True, help='預測結果 CSV (需包含 label, pred)')
    
    args = parser.parse_args()

    # 修正 1: 使用 r'' (Raw String) 來處理 Windows 路徑，避免 \U 被誤判
    # map_file_path = r'D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20191009\experiment_class_map.csv' # time variant 1
    map_file_path = r'D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20200219_20190611\experiment_class_map.csv' # time variant 2
    
    # 修正 2: 必須先呼叫 load_class_map 讀取檔案，轉成字典
    print(f"正在載入 Map: {map_file_path} ...")
    class_map = load_class_map(map_file_path)
    
    # 3. 計算誤差
    print(f"正在評估: {args.file} ...")
    result = calculate_mde(args.file, class_map)
    
    if result:
        print("\n" + "="*30)
        print(f"評估結果 (樣本數: {result['count']})")
        print("-" * 30)
        print(f"MDE (平均誤差):  {result['mde']:.4f} m")
        print(f"Std (標準差):    {result['std']:.4f} m")
        print(f"Median (中位數): {result['median']:.4f} m")
        print(f"95th Percentile: {result['p95']:.4f} m")
        print("="*30 + "\n")