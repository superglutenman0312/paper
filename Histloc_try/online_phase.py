import torch
import numpy as np
import os

# 從 HistLoc 的模組導入 (modules.py)
from modules import FeatureExtractor, LabelPredictor 
from data_loader_mall import get_mall_loaders # 使用更新後的 loader

# 從 HistLoc 的訓練腳本導入模型路徑
try:
    # ！！！ 注意：確保這裡導入的是 HistLoc 的 offline_phase ！！！
    from offline_phase import FE_MODEL_PATH_FINAL as FE_MODEL_PATH # 使用最終模型路徑
    from offline_phase import LP_MODEL_PATH_FINAL as LP_MODEL_PATH # 使用最終模型路徑
except ImportError:
    # 如果直接運行此檔案，提供預設路徑 (指向 HistLoc 的最終模型)
    # ！！確保這裡的路徑指向您訓練好的 HistLoc 模型！！
    FE_MODEL_PATH = 'checkpoints_histloc_10_1/histloc_mall_fe_final.pth' #<-- 指向 final
    LP_MODEL_PATH = 'checkpoints_histloc_10_1/histloc_mall_lp_final.pth' #<-- 指向 final
    # 或者您可以指向某個特定的 epoch，例如 epoch 200
    # FE_MODEL_PATH = 'checkpoints_histloc_10_1/histloc_mall_fe_epoch_200.pth'
    # LP_MODEL_PATH = 'checkpoints_histloc_10_1/histloc_mall_lp_epoch_200.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Online Phase (HistLoc MDE Evaluation for Mall) ---")
print(f"Using device: {device}")

def evaluate_histloc_mde():
    # --- 1. 獲取測試資料 (D2-test) 和 y_scaler ---
    # 傳入 target_train_ratio=1.0 避免 data_loader 打印抽取訊息
    loaders = get_mall_loaders(batch_size=32, target_train_ratio=1.0) 
    if loaders[0] is None:
        return
        
    # *** 關鍵修改：準備接收 6 個返回值 ***
    _, _, test_loader, y_scaler, num_beacons, _ = loaders #<-- 加了一個 _ 來接收 source_size
    
    if num_beacons == 0:
        print("錯誤：num_beacons 為 0。")
        return

    # --- 2. 初始化模型 (來自 modules.py) ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons, feature_dim=64).to(device)
    label_predictor = LabelPredictor(feature_dim=64, output_dim=2).to(device) # 迴歸版

    # --- 3. 載入校準後的權重 ---
    if not os.path.exists(FE_MODEL_PATH) or not os.path.exists(LP_MODEL_PATH):
        print(f"錯誤：找不到模型檔案 '{FE_MODEL_PATH}' 或 '{LP_MODEL_PATH}'。")
        print("請確保路徑正確，並已執行 'python offline_phase.py' (HistLoc 版本)。")
        return

    # 加入 map_location
    feature_extractor.load_state_dict(torch.load(FE_MODEL_PATH, map_location=device))
    label_predictor.load_state_dict(torch.load(LP_MODEL_PATH, map_location=device))
    print(f"成功從以下路徑載入模型權重：")
    print(f"  FE: {FE_MODEL_PATH}")
    print(f"  LP: {LP_MODEL_PATH}")

    feature_extractor.eval()
    label_predictor.eval()

    all_predictions_scaled = []
    all_true_scaled = []

    # --- 4. 執行預測 ---
    with torch.no_grad():
        for X_batch, y_batch_scaled in test_loader:
            X_batch = X_batch.to(device)
            features = feature_extractor(X_batch)
            predictions_scaled = label_predictor(features) 
            all_predictions_scaled.append(predictions_scaled.cpu().numpy())
            all_true_scaled.append(y_batch_scaled.cpu().numpy())

    # --- 5. 還原座標並計算 MDE ---
    all_predictions_scaled = np.concatenate(all_predictions_scaled)
    all_true_scaled = np.concatenate(all_true_scaled)
    all_predictions_meters = y_scaler.inverse_transform(all_predictions_scaled)
    all_true_meters = y_scaler.inverse_transform(all_true_scaled)
    errors_squared = np.sum((all_predictions_meters - all_true_meters)**2, axis=1)
    # 加入 np.maximum 防止 sqrt 報錯
    distances = np.sqrt(np.maximum(errors_squared, 1e-12)) 
    mde = np.mean(distances)
    
    print("\n--- 預測結果 (Online Phase) ---")
    print(f"測試資料 (D2-test) 筆數: {len(distances)}")
    print(f"平均距離誤差 (MDE): {mde:.4f} 公尺")
    
    print("\n前 5 筆預測 (單位：公尺):")
    print("   [預測 X, Y]  <-->  [真實 X, Y]   (誤差)")
    # 加入 min() 防止索引超出範圍
    for i in range(min(5, len(distances))): 
        print(f"   [{all_predictions_meters[i][0]:.2f}, {all_predictions_meters[i][1]:.2f}] <--> [{all_true_meters[i][0]:.2f}, {all_true_meters[i][1]:.2f}]   ({distances[i]:.2f}m)")

if __name__ == "__main__":
    evaluate_histloc_mde()