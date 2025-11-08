import pandas as pd
import numpy as np
import argparse
import sys

class SimpleEvaluator:
    """
    一個簡化版的評估器，專注於計算 MDE (Mean Distance Error)。
    
    此版本 "硬編碼" 了 MCSL (41 RPs) 資料集的網格佈局，
    使用 class_to_coordinate 函數來計算物理距離。
    
    它不需要外部的 ground_truth.csv 檔案。
    """

    def __init__(self):
        """
        初始化評S器。
        [改編自: evaluator.py, lines 21-22]
        """
        # (移除 ground_truth_map 的載入)
        self.results_df = None
        
        # [來自: evaluator.py, line 113]
        # MCSL (41 RPs) 的網格佈局表
        self.coord_table = np.array([
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
        ], dtype = int)

    def class_to_coordinate(self, rp_label):
        """
        將 RP 標籤 (1-41) 轉換為網格索引 [x, y]。
        [來自: evaluator.py, lines 112-124]
        """
        try:
            position = np.argwhere(self.coord_table == rp_label)
            if position.size == 0:
                print(f"警告: 在座標表中找不到 RP {rp_label}。返回 [0, 0]。")
                return [0, 0]
            
            y_grid = position[0][0]
            x_grid = position[0][1]
            coordinate = [x_grid, y_grid]
            return coordinate
        except Exception as e:
            print(f"錯誤: 轉換 RP {rp_label} 時出錯: {e}。返回 [0, 0]。")
            return [0, 0]

    def euclidean_distance(self, p1_grid, p2_grid):
        """
        計算兩個網格索引 [x, y] 之間的物理距離 (公尺)。
        網格間距為 0.6 公尺。
        [來自: evaluator.py, lines 126-127]
        """
        delta_x = p1_grid[0] - p2_grid[0]
        delta_y = p1_grid[1] - p2_grid[1]
        
        # 將網格距離轉換為物理距離 (公尺)
        return np.sqrt(delta_x**2 + delta_y**2) * 0.6

    def calculate_distance_error(self, predictions_df):
        """
        計算每筆預測的歐幾里得距離誤差。
        [改編自: evaluator.py, lines 43-55]
        """
        
        # --- 核心邏輯 (使用硬編碼轉換) ---
        
        # 1. 轉換 'label' (真實 RP) -> (X_grid_true, Y_grid_true)
        predictions_df['coord_true'] = predictions_df['label'].apply(self.class_to_coordinate)
        
        # 2. 轉換 'pred' (預測 RP) -> (X_grid_pred, Y_grid_pred)
        predictions_df['coord_pred'] = predictions_df['pred'].apply(self.class_to_coordinate)
        
        # 3. 計算物理距離 (公尺)
        predictions_df['Distance_Error'] = predictions_df.apply(
            lambda row: self.euclidean_distance(row['coord_true'], row['coord_pred']),
            axis=1
        )
        
        self.results_df = predictions_df

    def calculate_mde(self):
        """
        計算平均距離誤差 (Mean Distance Error)。
        [來自: evaluator.py, lines 57-60]
        """
        if self.results_df is None or 'Distance_Error' not in self.results_df.columns:
            print("錯誤: 尚未計算距離誤差。", file=sys.stderr)
            return 0.0
        
        return self.results_df['Distance_Error'].mean()

# --- 主程式執行區 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='(簡化版) 計算預測結果的 MDE (使用 MCSL 41-RPs 硬編碼座標)。')
    parser.add_argument(
        '--predictions', 
        type=str, 
        required=True, 
        help='預測結果的 CSV 檔案路徑 (必須包含 "label" 和 "pred" 欄位)'
    )
    # [MODIFIED] 移除了 --ground_truth 參數
    
    args = parser.parse_args()

    # 1. 載入預測結果檔
    try:
        predictions_df = pd.read_csv(args.predictions)
        if 'label' not in predictions_df.columns or 'pred' not in predictions_df.columns:
            print(f"錯誤: 預測檔案 '{args.predictions}' 必須包含 'label' 和 'pred' 欄位。", file=sys.stderr)
        else:
            # 2. 初始化評估器
            evaluator = SimpleEvaluator()
            
            # 3. 計算距離誤差
            evaluator.calculate_distance_error(predictions_df)
            
            # 4. 計算 MDE
            mde = evaluator.calculate_mde()
            
            print("\n--- MDE 計算完成 (MCSL 41-RPs) ---")
            print(f"  預測檔案: {args.predictions}")
            print(f"\n  平均距離誤差 (MDE): {mde:.4f} (公尺)")

    except FileNotFoundError:
        print(f"錯誤: 找不到預測檔案: {args.predictions}", file=sys.stderr)
    except Exception as e:
        print(f"處理預測檔案時發生錯誤: {e}", file=sys.stderr)