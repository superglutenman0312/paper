import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# 從 DANN 的模組導入
from modules_dann import FeatureExtractor, LabelPredictor, DomainDiscriminator, GradientReversalLayer
from data_loader_mall import get_mall_loaders # 使用共用的 loader

# --- 1. 設置訓練參數 ---
BATCH_SIZE = 32
LEARNING_RATE_G = 0.0001 
LEARNING_RATE_C = 0.0001 
EPOCHS = 200 # 保持較高 Epochs
# DANN 的 GRL alpha 會動態調整，不需要 Beta_Adv
GAMMA_L_T = 1.0    # *** Target 標籤損失 L_y_t 的權重 (設為 1.0 代表同等重要) ***
CRITIC_ITERATIONS = 1 # *** 注意：DANN 通常 G 和 D 交替訓練 1:1，不像 WDGRL 需要多次 Critic 更新 ***
TARGET_RATIO_INV = 10 # Source:Target = 10:1

# 模型儲存路徑
CHECKPOINT_DIR = 'checkpoints_dann_10_1_sda' # 新資料夾
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
FE_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'dann_mall_fe_final.pth')
LP_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'dann_mall_lp_final.pth')
# Discriminator 在預測時不需要

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Offline Phase (DANN Transfer Learning for Mall - SDA 10:1) ---")
print(f"Using device: {device}")

def train_dann_regression_sda_10_1():
    # --- 2. 載入資料 (加入比例控制邏輯) ---
    temp_loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=1.0)
    if temp_loaders[0] is None: return
    _, temp_target_loader, _, _, _, source_size = temp_loaders
    full_target_train_size = len(temp_target_loader.dataset)
    del temp_loaders, temp_target_loader

    target_subset_size = source_size / TARGET_RATIO_INV
    # 確保 ratio 不超過 1.0
    target_train_actual_ratio = min(1.0, target_subset_size / full_target_train_size) 
    
    print(f"Source size: {source_size}, Full Target train size: {full_target_train_size}")
    print(f"目標 Target 子集大小: {int(source_size / TARGET_RATIO_INV)}, 實際比例: {target_train_actual_ratio:.4f}")

    loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=target_train_actual_ratio)
    if loaders[0] is None: return
    source_loader, target_loader, _, _, num_beacons, _ = loaders
    print(f"Num Beacons (APs): {num_beacons}")
    print(f"Actual Target train subset size used: {len(target_loader.dataset)}")

    # --- 3. 初始化模型 (不變) ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons, feature_dim=64).to(device) # G_f
    label_predictor = LabelPredictor(feature_dim=64, output_dim=2).to(device)     # G_y
    domain_discriminator = DomainDiscriminator(feature_dim=64).to(device)        # G_d
    grl = GradientReversalLayer() 
    feature_extractor.train(); label_predictor.train(); domain_discriminator.train()

    # --- 4. 設置優化器和損失函數 (不變) ---
    params_to_optimize = (
        list(feature_extractor.parameters()) +
        list(label_predictor.parameters()) +
        list(domain_discriminator.parameters())
    )
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE_G, betas=(0.5, 0.9)) # DANN 通常用 Adam 或 SGD
    
    criterion_label = nn.MSELoss() # L_y 使用 MSE
    criterion_domain = nn.CrossEntropyLoss() # L_d 使用 CrossEntropy

    print(f"開始訓練 {EPOCHS} 個 Epochs...")
    print(f"Gamma L_t (Target Label Loss Weight): {GAMMA_L_T}")

    domain_labels_s = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
    domain_labels_t = torch.ones(BATCH_SIZE, dtype=torch.long).to(device)
    domain_labels_combined = torch.cat((domain_labels_s, domain_labels_t), dim=0)

    # --- 5. 執行訓練迴圈 (DANN - **SDA** 邏輯) ---
    len_dataloader = min(len(source_loader), len(target_loader))
    if len_dataloader == 0: return
        
    for epoch in range(EPOCHS):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        avg_total_loss, avg_ly_s, avg_ly_t, avg_ld = 0, 0, 0, 0
        
        # *** 修改：迴圈次數由較短的 target_loader 決定 ***
        for i in range(len(target_loader)): # Iterate based on target loader
            # 動態調整 GRL alpha
            p = float(epoch * len(target_loader) + i) / float(EPOCHS * len(target_loader))
            grl_alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 確保 source loader 也能循環獲取數據
            try: X_s_batch, y_s_batch = next(source_iter)
            except StopIteration: source_iter = iter(source_loader); X_s_batch, y_s_batch = next(source_iter)
            
            # 從 target loader 獲取數據 (包含標籤)
            X_t_batch, y_t_batch = next(target_iter)

            X_s_batch, y_s_batch = X_s_batch.to(device), y_s_batch.to(device)
            X_t_batch, y_t_batch = X_t_batch.to(device), y_t_batch.to(device)

            optimizer.zero_grad()

            # --- DANN-SDA 核心邏輯 ---
            # 1. 特徵提取 (G_f)
            features_s = feature_extractor(X_s_batch)
            features_t = feature_extractor(X_t_batch)
            
            # 2. 標籤預測損失 (L_y) - *** SDA 修改：同時計算 Source 和 Target ***
            predictions_s = label_predictor(features_s)
            loss_y_s = criterion_label(predictions_s, y_s_batch)
            
            predictions_t = label_predictor(features_t)
            loss_y_t = criterion_label(predictions_t, y_t_batch)
            
            # 合併標籤損失 (可以加權重)
            loss_y_combined = loss_y_s + (GAMMA_L_T * loss_y_t)
            
            # 3. 域判別損失 (L_d)
            features_combined = torch.cat((features_s, features_t), dim=0)
            features_reversed = grl(features_combined, grl_alpha) # 應用 GRL
            domain_predictions = domain_discriminator(features_reversed) 
            loss_d = criterion_domain(domain_predictions, domain_labels_combined)
            
            # 4. 總損失 (同時包含 L_y_s, L_y_t 和 L_d)
            loss_total = loss_y_combined + loss_d
            
            # 5. 反向傳播
            loss_total.backward()
            optimizer.step()
            # --- DANN-SDA 邏輯結束 ---
            
            avg_total_loss += loss_total.item()
            avg_ly_s += loss_y_s.item()
            avg_ly_t += loss_y_t.item()
            avg_ld += loss_d.item()

        # *** 修改：用 len(target_loader) 計算平均 ***
        avg_total_loss /= len(target_loader); avg_ly_s /= len(target_loader)
        avg_ly_t /= len(target_loader); avg_ld /= len(target_loader)
        
        print_interval = max(1, EPOCHS // 20) 
        if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == EPOCHS -1:
            rmse_s = np.sqrt(avg_ly_s)
            rmse_t = np.sqrt(avg_ly_t)
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}], GRL_Alpha: {grl_alpha:.3f}, L_total: {avg_total_loss:.4f}, RMSE_S: {rmse_s:.4f}, RMSE_T: {rmse_t:.4f}, L_d: {avg_ld:.4f}")

        # 每 10 個 Epoch 儲存一次模型
        if (epoch + 1) % 10 == 0:
            fe_path = os.path.join(CHECKPOINT_DIR, f'dann_mall_fe_epoch_{epoch+1}.pth')
            lp_path = os.path.join(CHECKPOINT_DIR, f'dann_mall_lp_epoch_{epoch+1}.pth')
            torch.save(feature_extractor.state_dict(), fe_path)
            torch.save(label_predictor.state_dict(), lp_path)

    # --- 6. 儲存最終模型權重 (不變) ---
    torch.save(feature_extractor.state_dict(), FE_MODEL_PATH_FINAL)
    torch.save(label_predictor.state_dict(), LP_MODEL_PATH_FINAL)
    print(f"訓練完成。最終模型已儲存。Checkpoints 於 '{CHECKPOINT_DIR}/'")

if __name__ == "__main__":
    train_dann_regression_sda_10_1()