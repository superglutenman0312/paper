import pandas as pd
import numpy as np
# import pickle  <- 已移除，不再需要
import argparse
import os

# --- 1. (已移除) 'label_map.pkl' 的路徑設定 ---
# (不再需要從外部檔案讀取)

# --- 2. 設定固定的 標籤/座標 對照表 ---
# (此處替換為您提供的字典)
label_to_coordinate = {
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
print("已使用內建的 標籤/座標 對照表。")


# --- 3. 主程式 ---
if __name__ == "__main__":
    
    # 設定命令列參數
    parser = argparse.ArgumentParser(description='Calculate MDE from a HistLoc prediction CSV file.')
    parser.add_argument('--file', type=str, required=True, 
                          help='Path to the prediction CSV file produced by the model.')
    
    args = parser.parse_args()

    # 檢查預測檔案是否存在
    if not os.path.exists(args.file):
        print(f"錯誤：找不到預測檔案: {args.file}")
        exit()

    print(f"正在讀取預測檔案: {args.file}")
    
    # 讀取預測 CSV 檔案
    try:
        results = pd.read_csv(args.file)
        # 檢查必要的欄位是否存在 (根據 evaluator.py)
        if 'label' not in results.columns or 'pred' not in results.columns:
            print(f"錯誤：CSV 檔案必須包含 'label' 和 'pred' 欄位。")
            print(f"   (您說的 'a,b' 欄位名稱可能需要在此程式碼中修改)")
            exit()
    except Exception as e:
        print(f"讀取 CSV 檔案時發生錯誤: {e}")
        exit()

    # --- 4. 計算 MDE ---
    errors = []
    # 遍歷 CSV 中的每一行
    for idx, row in results.iterrows():
        try:
            # 確保標籤是整數型態，以匹配字典的 'key'
            pred_label = int(row['pred'])      # 取得預測的標籤
            actual_label = int(row['label'])   # 取得真實的標籤
            
            # 將標籤轉換為 (x, y) 座標
            pred_coord = label_to_coordinate[pred_label]
            actual_coord = label_to_coordinate[actual_label]
            
            # 計算歐幾里得距離 (Euclidean distance)
            distance_error = np.linalg.norm(np.array(pred_coord) - np.array(actual_coord))
            
            errors.append(distance_error)
        
        except KeyError as e:
            # 如果 'label' 或 'pred' 中的標籤在 內建字典 中找不到
            print(f"警告：在 座標對照表 中找不到標籤 {e}。跳過此行。")
        except Exception as e:
            print(f"處理第 {idx} 行時發生錯誤: {e}。跳過此行。")

    # --- 5. 顯示結果 ---
    if errors:
        # 計算平均距離誤差
        mean_distance_error = np.mean(errors) 
        
        print("\n--- MDE 計算完成 ---")
        print(f"總共計算了 {len(errors)} 筆預測")
        print(f"平均距離誤差 (MDE): {mean_distance_error:.4f} 公尺")
    else:
        print("錯誤：沒有成功計算任何誤差。")