import torch
import torch.nn as nn
import torch.optim as optim
import os

# 從我們建立的檔案中導入
from modules import FeatureExtractor, LabelPredictor, compute_histogram_loss
from data_generator import get_data_loaders

# --- 1. 設置訓練參數 (基於論文 Table 3.2 [cite: 408]) ---
# 模擬資料參數
NUM_BEACONS = 168  # 類似 UM DSI DB
NUM_CLASSES = 49   # 類似 UM DSI DB
N_SOURCE = 2000
N_TARGET_LABELED = 200 # 模擬 10:1 的情況 [cite: 460]

# 訓練參數
BATCH_SIZE = 32     # [cite: 408]
LEARNING_RATE = 0.001 # [cite: 408]
EPOCHS = 10        # 論文設 500 [cite: 408]，這裡設 100 用於快速演示
ALPHA = 0.1         # [cite: 408]
BETA = 10           # [cite: 408]
UNLABELED_TARGET = False # 設為 True 來模擬 3.1.3 節的情境 [cite: 377]

# 模型儲存路徑
FE_MODEL_PATH = 'feature_extractor.pth'
LP_MODEL_PATH = 'label_predictor.pth'

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Offline Phase (Algorithm 1) ---")
print(f"Using device: {device}")

def train_histloc():
    # --- 2. 載入資料 ---
    source_loader, target_loader, _ = get_data_loaders(
        num_beacons=NUM_BEACONS,
        num_classes=NUM_CLASSES,
        n_source=N_SOURCE,
        n_target=N_TARGET_LABELED,
        batch_size=BATCH_SIZE
    )

    # --- 3. 初始化模型 ---
    feature_extractor = FeatureExtractor(input_dim=NUM_BEACONS).to(device)
    label_predictor = LabelPredictor(num_classes=NUM_CLASSES).to(device)
    
    feature_extractor.train()
    label_predictor.train()

    # --- 4. 設置優化器和損失函數 ---
    params_to_optimize = list(feature_extractor.parameters()) + list(label_predictor.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE) # 使用 Adam [cite: 408]
    
    criterion_label = nn.CrossEntropyLoss() # 使用交叉熵 [cite: 325]

    print(f"開始訓練... (Unlabeled Target: {UNLABELED_TARGET})")

    # --- 5. 執行訓練迴圈 (Algorithm 1) [cite: 339-375] ---
    for epoch in range(EPOCHS):
        target_iter = iter(target_loader)
        
        for i, (X_s_batch, y_s_batch) in enumerate(source_loader):
            
            try:
                X_t_batch, y_t_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                X_t_batch, y_t_batch = next(target_iter)

            X_s_batch, y_s_batch = X_s_batch.to(device), y_s_batch.to(device)
            X_t_batch, y_t_batch = X_t_batch.to(device), y_t_batch.to(device)

            optimizer.zero_grad()

            # 2. 特徵提取 (f_F)
            features_s = feature_extractor(X_s_batch) # X_C^(S) [cite: 364]
            features_t = feature_extractor(X_t_batch) # X_C^(T) [cite: 364]

            # 3. 計算域判別器損失 (L_D)
            loss_d = compute_histogram_loss(features_s, features_t) # [cite: 366-368]

            # 4. 標籤預測 (f_L) & 計算 L_L
            if UNLABELED_TARGET:
                # 情境 3.1.3: 僅使用源域標籤 [cite: 381-383]
                predictions = label_predictor(features_s)
                loss_l = criterion_label(predictions, y_s_batch)
            else:
                # 情境 3.1.2: 使用源域和目標域標籤 [cite: 370-372]
                features_combined = torch.cat((features_s, features_t), dim=0) 
                labels_combined = torch.cat((y_s_batch, y_t_batch), dim=0)
                predictions = label_predictor(features_combined) 
                loss_l = criterion_label(predictions, labels_combined) 
            
            # 5. 計算總損失 (L_total)
            loss_total = (ALPHA * loss_d) + (BETA * loss_l) # [cite: 333, 373]

            # 6. 反向傳播與優化
            loss_total.backward() # [cite: 374]
            optimizer.step()

        if (epoch + 1) % (EPOCHS // 10) == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], L_total: {loss_total.item():.4f}, L_D: {loss_d.item():.4f}, L_L: {loss_l.item():.4f}")

    # --- 6. 儲存模型權重 ---
    torch.save(feature_extractor.state_dict(), FE_MODEL_PATH)
    torch.save(label_predictor.state_dict(), LP_MODEL_PATH)
    print(f"訓練完成。模型已儲存至 '{FE_MODEL_PATH}' 和 '{LP_MODEL_PATH}'")

if __name__ == "__main__":
    train_histloc()