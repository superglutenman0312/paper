import torch
import numpy as np
import os

# 從我們建立的檔案中導入
from modules import FeatureExtractor, LabelPredictor
from data_generator import get_data_loaders

# 嘗試從 offline_phase 導入參數，如果失敗 (例如直接執行此檔)，則使用預設值
try:
    from offline_phase import NUM_BEACONS, NUM_CLASSES, N_SOURCE, N_TARGET_LABELED, BATCH_SIZE, FE_MODEL_PATH, LP_MODEL_PATH
except ImportError:
    print("警告：無法從 offline_phase 導入參數，使用預設值。")
    NUM_BEACONS = 168
    NUM_CLASSES = 49
    N_SOURCE = 2000
    N_TARGET_LABELED = 200
    BATCH_SIZE = 32
    FE_MODEL_PATH = 'feature_extractor.pth'
    LP_MODEL_PATH = 'label_predictor.pth'


# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Online Phase (Algorithm 2) ---")
print(f"Using device: {device}")

def predict_histloc():
    # --- 1. 獲取測試資料和 Scaler ---
    # 再次調用 data_generator 來獲取 "相同" 的模擬測試資料和 scaler
    # 在真實應用中，您會載入已儲存的 scaler 和真實的測試資料
    _, _, test_data_package = get_data_loaders(
        num_beacons=NUM_BEACONS,
        num_classes=NUM_CLASSES,
        n_source=N_SOURCE,
        n_target=N_TARGET_LABELED,
        batch_size=BATCH_SIZE
    )
    
    X_test_rssi, y_test_labels, scaler_t = test_data_package

    # --- 2. 初始化模型 ---
    feature_extractor = FeatureExtractor(input_dim=NUM_BEACONS).to(device)
    label_predictor = LabelPredictor(num_classes=NUM_CLASSES).to(device)

    # --- 3. 載入校準後的權重 ---
    # [cite_start]對應 Algorithm 2 的 [cite: 401, 402]
    if not os.path.exists(FE_MODEL_PATH) or not os.path.exists(LP_MODEL_PATH):
        print(f"錯誤：找不到模型檔案 '{FE_MODEL_PATH}' 或 '{LP_MODEL_PATH}'。")
        print("請先執行 'python offline_phase.py' 進行訓練。")
        return

    feature_extractor.load_state_dict(torch.load(FE_MODEL_PATH))
    label_predictor.load_state_dict(torch.load(LP_MODEL_PATH))
    print(f"成功從 '{FE_MODEL_PATH}' 和 '{LP_MODEL_PATH}' 載入模型權重。")

    feature_extractor.eval()
    label_predictor.eval()

    # [cite_start]--- 4. 執行預測 (Algorithm 2) [cite: 397-406] ---
    
    # [cite_start]4.1 標準化 [cite: 400]
    X_test_normalized = scaler_t.transform(X_test_rssi)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # [cite_start]4.2 特徵提取 [cite: 403]
        features_c = feature_extractor(X_test_tensor)
        # [cite_start]4.3 標籤預測 [cite: 404]
        predictions = label_predictor(features_c)
        # [cite_start]4.4 獲取 RP [cite: 405]
        predicted_rps = torch.argmax(predictions, dim=1)
        all_predictions = predicted_rps.cpu().numpy()

    # --- 5. 顯示結果 ---
    correct_count = np.sum(all_predictions == y_test_labels)
    accuracy = (correct_count / len(y_test_labels)) * 100
    
    print("\n--- 預測結果 (Online Phase) ---")
    print(f"真實 Target RPs (前20筆): {y_test_labels[:20]}")
    print(f"預測 Target RPs (前20筆): {all_predictions[:20]}")
    print(f"\n在 {len(y_test_labels)} 筆目標域測試資料上的準確度: {accuracy:.2f}%")

if __name__ == "__main__":
    predict_histloc()