import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# 從我們建立的檔案中導入
from modules import FeatureExtractor, LabelPredictor
from data_loader import get_loaders

# --- 1. 設置訓練參數 ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2000 # 迴歸任務可能需要更多 Epochs 來收斂

# 模型儲存路徑
FE_MODEL_PATH = 'feature_extractor.pth'
LP_MODEL_PATH = 'label_predictor.pth'

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 開始訓練 (階段一：迴歸模型) ---")
print(f"Using device: {device}")

def train_regression_model():
    # --- 2. 載入資料 ---
    train_loader, _, _, num_beacons = get_loaders(BATCH_SIZE)
    
    if train_loader is None:
        return # 載入資料失敗

    # --- 3. 初始化模型 ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons).to(device)
    label_predictor = LabelPredictor(output_dim=2).to(device) # 輸出維度為 2
    
    feature_extractor.train()
    label_predictor.train()

    # --- 4. 設置優化器和損失函數 ---
    params_to_optimize = list(feature_extractor.parameters()) + list(label_predictor.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
    
    # *** 關鍵修改：使用 MSELoss (均方誤差) 進行迴歸 ***
    criterion = nn.MSELoss()

    print(f"開始訓練 {EPOCHS} 個 Epochs...")

    # --- 5. 執行訓練迴圈 (單一 domain) ---
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # 前向傳播
            features = feature_extractor(X_batch)
            predictions = label_predictor(features)

            # 計算損失 (y_batch 的 shape 是 [32, 2], predictions 也是 [32, 2])
            loss = criterion(predictions, y_batch)
            
            # 反向傳播與優化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # 顯示 MSE Loss。開根號 (RMSE) 會更接近公尺誤差
            print(f"Epoch [{epoch+1}/{EPOCHS}], Avg. MSE Loss: {avg_loss:.6f}, (RMSE: {np.sqrt(avg_loss):.4f})")

    # --- 6. 儲存模型權重 ---
    torch.save(feature_extractor.state_dict(), FE_MODEL_PATH)
    torch.save(label_predictor.state_dict(), LP_MODEL_PATH)
    print(f"訓練完成。模型已儲存至 '{FE_MODEL_PATH}' 和 '{LP_MODEL_PATH}'")

if __name__ == "__main__":
    train_regression_model()