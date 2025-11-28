import pandas as pd
import numpy as np
import argparse
import os

# --- 1. (已移除) 舊的 'label_to_coordinate' 字典 ---

# --- 2. (新增) 'Evaluator' class 中的「標籤/座標 對照表」 ---
# 這是你提供的 table 查表算法
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
    
    # 使用 np.argwhere 查找標籤 'a' 在 table 中的位置
    locations = np.argwhere(table == a)
    
    # 錯誤處理：如果找不到標籤
    if locations.size == 0:
        raise KeyError(f"標籤 {a} 在 table 中找不到。")
        
    x = locations[0][1]  # 欄索引 (column index)
    y = locations[0][0]  # 列索引 (row index)
    coordinate = [x, y]
    return coordinate

# --- 3. (新增) 'Evaluator' class 中的「距離計算公式」 ---
def euclidean_distance(p1, p2):
    """
    計算歐幾里得距離，並乘以 0.6 的縮放因子。
    p1, p2 均為 [x, y] 格式的座標。
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) * 0.6

# ---
print("已使用 'class_to_coordinate' (格點 table) 算法及 0.6 縮放因子。")
# ---


# --- 4. 主程式 (沿用 evaluator.py 的架構) ---
if __name__ == "__main__":
    
    # (保留) 設定命令列參數
    parser = argparse.ArgumentParser(description='Calculate MDE from a prediction CSV file.')
    parser.add_argument('--file', type=str, required=True, 
                          help='Path to the prediction CSV file produced by the model.')
    
    args = parser.parse_args()

    # (保留) 檢查預測檔案是否存在
    if not os.path.exists(args.file):
        print(f"錯誤：找不到預測檔案: {args.file}")
        exit()

    print(f"正在讀取預測檔案: {args.file}")
    
    # (保留) 讀取預測 CSV 檔案
    try:
        results = pd.read_csv(args.file)
        if 'label' not in results.columns or 'pred' not in results.columns:
            print(f"錯誤：CSV 檔案必須包含 'label' 和 'pred' 欄位。")
            exit()
    except Exception as e:
        print(f"讀取 CSV 檔案時發生錯誤: {e}")
        exit()

    # --- 5. (修改) 計算 MDE (使用新的函式) ---
    errors = []
    # 遍歷 CSV 中的每一行
    for idx, row in results.iterrows():
        try:
            # 確保標籤是整數型態
            pred_label = int(row['pred'])      # 取得預測的標籤
            actual_label = int(row['label'])   # 取得真實的標籤
            
            # (修改) 使用新的 'table' 查表法
            pred_coord = class_to_coordinate(pred_label)
            actual_coord = class_to_coordinate(actual_label)
            
            # (修改) 使用新的 'euclidean_distance' (含 0.6 縮放)
            distance_error = euclidean_distance(pred_coord, actual_coord)
            
            errors.append(distance_error)
        
        except KeyError as e:
            # 如果 'label' 或 'pred' 中的標籤在 table 中找不到
            print(f"警告：{e}。跳過此行。")
        except Exception as e:
            print(f"處理第 {idx} 行時發生錯誤: {e}。跳過此行。")

    # --- 6. (保留) 顯示結果 ---
    if errors:
        # 計算平均距離誤差
        mean_distance_error = np.mean(errors) 
        
        print("\n--- MDE 計算完成 ---")
        print(f"總共計算了 {len(errors)} 筆預測")
        # (修改) 標註 MDE 使用的算法
        print(f"平均距離誤差 (MDE) (使用 0.6 縮放): {mean_distance_error:.4f} 公尺")
    else:
        print("錯誤：沒有成功計算任何誤差。")