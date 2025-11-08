import pandas as pd
import numpy as np
import pickle
import argparse
import os

# --- 1. 設定 'label_map.pkl' 的路徑 ---
# (請根據您的 'label_map.pkl' 檔案實際位置修改此路徑)
# 根據您的 DANN_CORR_MALL.py 指令，我猜測您的 mall_data 資料夾在 'Histloc_try' 內
LABEL_MAP_PATH = 'D:/paper_thesis/Histloc_try/mall_data/Mall/label_map.pkl' 

# --- 2. 載入並反轉座標/標籤對照表 ---
try:
    # 載入 { 座標(tuple): 標籤(int) }
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f) 
    # 反轉字典，變為 { 標籤(int): 座標(tuple) }
    label_to_coordinate = {value: key for key, value in label_map.items()} 
    print(f"成功載入座標對照表: {LABEL_MAP_PATH}")
except FileNotFoundError:
    print(f"錯誤：找不到 'label_map.pkl' 檔案於: {LABEL_MAP_PATH}")
    print("請檢查 `eval.py` 檔案中的 LABEL_MAP_PATH 變數是否設定正確。")
    exit()
except Exception as e:
    print(f"載入 {LABEL_MAP_PATH} 時發生錯誤: {e}")
    exit()


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
            pred_label = row['pred']        # 取得預測的標籤
            actual_label = row['label']     # 取得真實的標籤
            
            # 將標籤轉換為 (x, y) 座標
            pred_coord = label_to_coordinate[pred_label]
            actual_coord = label_to_coordinate[actual_label]
            
            # 計算歐幾里得距離 (Euclidean distance)
            distance_error = np.linalg.norm(np.array(pred_coord) - np.array(actual_coord))
            
            errors.append(distance_error)
        
        except KeyError as e:
            # 如果 'label' 或 'pred' 中的標籤在 .pkl 檔案中找不到
            print(f"警告：在 label_map 中找不到標籤 {e}。跳過此行。")
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