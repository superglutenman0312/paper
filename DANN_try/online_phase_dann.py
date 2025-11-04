import torch
import numpy as np
import os

# 從 DANN 的模組導入
from modules_dann import FeatureExtractor, LabelPredictor
from data_loader_mall import get_mall_loaders # 使用共用的 loader

# 從 DANN (SDA 10:1) 的訓練腳本導入模型路徑
try:
    # ！！！ 注意：確保這裡導入的是 DANN 的 offline_phase ！！！
    from offline_phase_dann import FE_MODEL_PATH_FINAL as FE_MODEL_PATH # 使用最終模型路徑
    from offline_phase_dann import LP_MODEL_PATH_FINAL as LP_MODEL_PATH # 使用最終模型路徑
except ImportError:
    # 如果直接運行此檔案，提供預設路徑 (指向 DANN SDA 10:1 的最終模型)
    # ！！確保這裡的路徑指向您訓練好的 DANN SDA 10:1 模型！！
    FE_MODEL_PATH = 'checkpoints_dann_10_1_sda/dann_mall_fe_final.pth' #<-- 指向 final
    LP_MODEL_PATH = 'checkpoints_dann_10_1_sda/dann_mall_lp_final.pth' #<-- 指向 final
    # 或者您可以指向某個特定的 epoch
    # FE_MODEL_PATH = 'checkpoints_dann_10_1_sda/dann_mall_fe_epoch_200.pth'
    # LP_MODEL_PATH = 'checkpoints_dann_10_1_sda/dann_mall_lp_epoch_200.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Online Phase (DANN SDA 10:1 MDE Evaluation for Mall) ---")
print(f"Using device: {device}")

def evaluate_dann_mde():
    # --- 1. 獲取測試資料 (D2-test) 和 y_scaler ---
    # 傳入 target_train_ratio=1.0 避免 data_loader 打印抽取訊息
    loaders = get_mall_loaders(batch_size=32, target_train_ratio=1.0) 
    if loaders[0] is None:
        return
        
    # 接收 6 個返回值
    _, _, test_loader, y_scaler, num_beacons, _ = loaders 
    
    if num_beacons == 0:
        print("錯誤：num_beacons 為 0。")
        return

    # --- 2. 初始化模型 (G_f, G_y) ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons, feature_dim=64).to(device)
    label_predictor = LabelPredictor(feature_dim=64, output_dim=2).to(device) # 迴歸

    # --- 3. 載入校準後的權重 ---
    if not os.path.exists(FE_MODEL_PATH) or not os.path.exists(LP_MODEL_PATH):
        print(f"錯誤：找不到模型檔案 '{FE_MODEL_PATH}' 或 '{LP_MODEL_PATH}'。")
        print("請確保路徑正確，並已執行 'python offline_phase_dann.py' (SDA 版本)。")
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

    # --- 4. 執行預測 (僅使用 G_f 和 G_y) ---
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
    distances = np.sqrt(np.maximum(errors_squared, 1e-12)) 
    mde = np.mean(distances)
    
    print("\n--- 預測結果 (Online Phase) ---")
    print(f"測試資料 (D2-test) 筆數: {len(distances)}")
    print(f"平均距離誤差 (MDE): {mde:.4f} 公尺")
    
    print("\n前 5 筆預測 (單位：公尺):")
    print("   [預測 X, Y]  <-->  [真實 X, Y]   (誤差)")
    for i in range(min(5, len(distances))): 
        print(f"   [{all_predictions_meters[i][0]:.2f}, {all_predictions_meters[i][1]:.2f}] <--> [{all_true_meters[i][0]:.2f}, {all_true_meters[i][1]:.2f}]   ({distances[i]:.2f}m)")

if __name__ == "__main__":
    evaluate_dann_mde()