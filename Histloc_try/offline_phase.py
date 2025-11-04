import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# 從我們建立的檔案中導入
from modules import FeatureExtractor, LabelPredictor, compute_histogram_loss 
from data_loader_mall import get_mall_loaders # 使用更新後的 loader

# --- 1. 設置訓練參數 ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200 
ALPHA = 0.1  # L_D 權重
BETA = 10    # L_L 權重
TARGET_RATIO_INV = 10 # 我們希望 Source:Target = 10:1

# 模型儲存路徑
CHECKPOINT_DIR = 'checkpoints_histloc_10_1' # 新資料夾
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
FE_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'histloc_mall_fe_final.pth')
LP_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'histloc_mall_lp_final.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Offline Phase (HistLoc Transfer Learning for Mall - Regression 10:1) ---")
print(f"Using device: {device}")

def train_histloc_regression_10_1():
    # --- 2. 載入資料 ---
    # 先載入一次以獲取 source_size
    temp_loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=1.0)
    if temp_loaders[0] is None: return
    _, temp_target_loader, _, _, _, source_size = temp_loaders
    full_target_train_size = len(temp_target_loader.dataset)
    del temp_loaders, temp_target_loader # 釋放記憶體

    # 計算目標比例
    target_subset_size = source_size / TARGET_RATIO_INV
    target_train_actual_ratio = target_subset_size / full_target_train_size
    
    print(f"Source size: {source_size}, Full Target train size: {full_target_train_size}")
    print(f"目標 Target 子集大小: {int(target_subset_size)}, 實際比例: {target_train_actual_ratio:.4f}")

    # 使用計算出的比例重新載入資料
    loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=target_train_actual_ratio)
    if loaders[0] is None: return
    source_loader, target_loader, _, _, num_beacons, _ = loaders # 忽略 source_size
    print(f"Num Beacons (APs): {num_beacons}")
    print(f"Actual Target train subset size used: {len(target_loader.dataset)}")

    # --- 3. 初始化模型 ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons, feature_dim=64).to(device)
    label_predictor = LabelPredictor(feature_dim=64, output_dim=2).to(device) 
    feature_extractor.train(); label_predictor.train()

    # --- 4. 設置優化器和損失函數 ---
    params_to_optimize = list(feature_extractor.parameters()) + list(label_predictor.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
    criterion_label = nn.MSELoss()

    print(f"開始訓練 {EPOCHS} 個 Epochs...")
    print(f"L_D 權重 (Alpha): {ALPHA}, L_L 權重 (Beta): {BETA}")

    # --- 5. 執行訓練迴圈 (不變) ---
    len_dataloader = min(len(source_loader), len(target_loader))
    if len_dataloader == 0:
        print("錯誤：Source 或 Target DataLoader 為空，請檢查 Batch Size 和資料子集大小。")
        return
        
    for epoch in range(EPOCHS):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        avg_total_loss, avg_ld, avg_ll_s, avg_ll_t = 0, 0, 0, 0
        
        # *** 修改：迴圈次數由較短的 target_loader 決定 ***
        for i in range(len(target_loader)): # Iter over target loader
            # 確保 source loader 也能循環獲取數據
            try: X_s_batch, y_s_batch = next(source_iter)
            except StopIteration: source_iter = iter(source_loader); X_s_batch, y_s_batch = next(source_iter)
            
            # 從 target loader 獲取數據
            X_t_batch, y_t_batch = next(target_iter)

            X_s_batch, y_s_batch = X_s_batch.to(device), y_s_batch.to(device)
            X_t_batch, y_t_batch = X_t_batch.to(device), y_t_batch.to(device)

            optimizer.zero_grad()
            features_s = feature_extractor(X_s_batch)
            features_t = feature_extractor(X_t_batch)
            loss_d = compute_histogram_loss(features_s, features_t) 
            features_combined = torch.cat((features_s, features_t), dim=0) 
            labels_combined = torch.cat((y_s_batch, y_t_batch), dim=0)
            predictions_combined = label_predictor(features_combined)
            loss_l = criterion_label(predictions_combined, labels_combined)
            loss_total = (ALPHA * loss_d) + (BETA * loss_l) 
            loss_total.backward()
            optimizer.step()
            
            with torch.no_grad():
                predictions_s = label_predictor(features_s); loss_l_s = criterion_label(predictions_s, y_s_batch)
                predictions_t = label_predictor(features_t); loss_l_t = criterion_label(predictions_t, y_t_batch)

            avg_total_loss += loss_total.item(); avg_ld += loss_d.item()
            avg_ll_s += loss_l_s.item(); avg_ll_t += loss_l_t.item()

        # *** 修改：用 len(target_loader) 計算平均 ***
        avg_total_loss /= len(target_loader); avg_ld /= len(target_loader)
        avg_ll_s /= len(target_loader); avg_ll_t /= len(target_loader)
        
        print_interval = max(1, EPOCHS // 20) 
        if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == EPOCHS -1:
            rmse_s = np.sqrt(avg_ll_s); rmse_t = np.sqrt(avg_ll_t)
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}], L_total: {avg_total_loss:.6f}, L_D: {avg_ld:.4f}, RMSE_S: {rmse_s:.4f}, RMSE_T: {rmse_t:.4f}")

        # 每 10 個 Epoch 儲存一次模型
        if (epoch + 1) % 10 == 0:
            fe_path = os.path.join(CHECKPOINT_DIR, f'histloc_mall_fe_epoch_{epoch+1}.pth')
            lp_path = os.path.join(CHECKPOINT_DIR, f'histloc_mall_lp_epoch_{epoch+1}.pth')
            torch.save(feature_extractor.state_dict(), fe_path)
            torch.save(label_predictor.state_dict(), lp_path)

    # --- 6. 儲存最終模型權重 (不變) ---
    torch.save(feature_extractor.state_dict(), FE_MODEL_PATH_FINAL)
    torch.save(label_predictor.state_dict(), LP_MODEL_PATH_FINAL)
    print(f"訓練完成。最終模型已儲存。Checkpoints 於 '{CHECKPOINT_DIR}/'")

if __name__ == "__main__":
    train_histloc_regression_10_1()