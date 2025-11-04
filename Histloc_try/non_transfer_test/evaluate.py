import torch
import numpy as np
import os

from modules import FeatureExtractor, LabelPredictor
from data_loader import get_loaders

# 嘗試從 train.py 導入模型路徑
try:
    from train import FE_MODEL_PATH, LP_MODEL_PATH
except ImportError:
    FE_MODEL_PATH = 'feature_extractor.pth'
    LP_MODEL_PATH = 'label_predictor.pth'

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 開始評估 (階段一：迴歸模型) ---")
print(f"Using device: {device}")

def evaluate_model():
    # --- 1. 獲取測試資料和 y_scaler (用於還原座標) ---
    _, test_loader, y_scaler, num_beacons = get_loaders(batch_size=32)
    
    if test_loader is None:
        return # 載入資料失敗
        
    if num_beacons == 0:
        print("錯誤：num_beacons 為 0，請檢查 data_loader.py")
        return

    # --- 2. 初始化模型 ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons).to(device)
    label_predictor = LabelPredictor(output_dim=2).to(device)

    # --- 3. 載入校準後的權重 ---
    if not os.path.exists(FE_MODEL_PATH) or not os.path.exists(LP_MODEL_PATH):
        print(f"錯誤：找不到模型檔案 '{FE_MODEL_PATH}' 或 '{LP_MODEL_PATH}'。")
        print("請先執行 'python train.py' 進行訓練。")
        return

    feature_extractor.load_state_dict(torch.load(FE_MODEL_PATH))
    label_predictor.load_state_dict(torch.load(LP_MODEL_PATH))
    print("成功載入模型權重。")

    feature_extractor.eval()
    label_predictor.eval()

    all_predictions_scaled = []
    all_true_scaled = []

    # --- 4. 執行預測 ---
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            features = feature_extractor(X_batch)
            predictions = label_predictor(features)
            
            all_predictions_scaled.append(predictions.cpu().numpy())
            all_true_scaled.append(y_batch.cpu().numpy())

    # --- 5. 還原座標並計算 MDE ---
    
    # 將所有批次合併成一個大陣列
    all_predictions_scaled = np.concatenate(all_predictions_scaled)
    all_true_scaled = np.concatenate(all_true_scaled)
    
    # *** 關鍵修改：使用 y_scaler 還原座標到「公尺」單位 ***
    all_predictions_meters = y_scaler.inverse_transform(all_predictions_scaled)
    all_true_meters = y_scaler.inverse_transform(all_true_scaled)
    
    # 計算每個點的歐幾里得距離誤差
    # (x_pred - x_true)^2 + (y_pred - y_true)^2
    errors_squared = np.sum((all_predictions_meters - all_true_meters)**2, axis=1)
    # sqrt(...)
    distances = np.sqrt(errors_squared)
    
    # 計算 MDE (Mean Distance Error)
    mde = np.mean(distances)
    
    print("\n--- 預測結果 (Online Phase) ---")
    print(f"測試資料筆數: {len(distances)}")
    print(f"平均距離誤差 (MDE): {mde:.4f} 公尺")
    
    # 顯示前幾筆預測
    print("\n前 5 筆預測 (單位：公尺):")
    print("   [預測 X, 預測 Y]  <-->  [真實 X, 真實 Y]   (誤差)")
    for i in range(5):
        print(f"   [{all_predictions_meters[i][0]:.2f}, {all_predictions_meters[i][1]:.2f}] <--> [{all_true_meters[i][0]:.2f}, {all_true_meters[i][1]:.2f}]   ({distances[i]:.2f}m)")

if __name__ == "__main__":
    evaluate_model()