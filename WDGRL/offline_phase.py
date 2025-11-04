import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# 從 WDGRL 模組導入
from modules_wdgrl import FeatureExtractor, LabelPredictor, DomainCritic, compute_gradient_penalty
from data_loader_mall import get_mall_loaders # 使用更新後的 loader

# --- 1. 設置訓練參數 ---
# (此部分與您提供的 offline_phase.py 相同)
BATCH_SIZE = 32
LEARNING_RATE_G = 0.0001 
LEARNING_RATE_C = 0.0001 
EPOCHS = 1000 # 保持較高 Epochs
LAMBDA_GP = 10     
BETA_ADV = 0.001   # 對抗損失權重 (使用您上次覺得可能較好的值)
GAMMA_L_T = 1.0    # *** 新增：Target 標籤損失 L_y_t 的權重 ***
CRITIC_ITERATIONS = 5 
TARGET_RATIO_INV = 10 # Source:Target = 10:1

# 模型儲存路徑
CHECKPOINT_DIR = 'checkpoints_wdgrl_10_1_sda' # 新資料夾
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
FE_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'wdgrl_mall_fe_final.pth')
LP_MODEL_PATH_FINAL = os.path.join(CHECKPOINT_DIR, 'wdgrl_mall_lp_final.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Offline Phase (WDGRL Transfer Learning for Mall - SDA 10:1) ---")
print(f"Using device: {device}")

def train_wdgrl_regression_sda_10_1():
    # --- 2. 載入資料 (與 HistLoc 相同邏輯) ---
    # (此部分與您提供的 offline_phase.py 相同)
    temp_loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=1.0)
    if temp_loaders[0] is None: return
    _, temp_target_loader, _, _, _, source_size = temp_loaders
    full_target_train_size = len(temp_target_loader.dataset)
    del temp_loaders, temp_target_loader

    target_subset_size = source_size / TARGET_RATIO_INV
    target_train_actual_ratio = target_subset_size / full_target_train_size
    
    print(f"Source size: {source_size}, Full Target train size: {full_target_train_size}")
    print(f"目標 Target 子集大小: {int(target_subset_size)}, 實際比例: {target_train_actual_ratio:.4f}")

    loaders = get_mall_loaders(BATCH_SIZE, target_train_ratio=target_train_actual_ratio)
    if loaders[0] is None: return
    source_loader, target_loader, _, _, num_beacons, _ = loaders
    print(f"Num Beacons (APs): {num_beacons}")
    print(f"Actual Target train subset size used: {len(target_loader.dataset)}")

    # --- 3. 初始化模型 (與您提供的 offline_phase.py 相同) ---
    feature_extractor = FeatureExtractor(input_dim=num_beacons, feature_dim=64).to(device)
    label_predictor = LabelPredictor(feature_dim=64, output_dim=2).to(device)
    domain_critic = DomainCritic(feature_dim=64).to(device)
    feature_extractor.train(); label_predictor.train(); domain_critic.train()

    # --- 4. 設置優化器和損失函數 (與您提供的 offline_phase.py 相同) ---
    optimizer_G = optim.Adam(list(feature_extractor.parameters()) + list(label_predictor.parameters()), lr=LEARNING_RATE_G, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(domain_critic.parameters(), lr=LEARNING_RATE_C, betas=(0.5, 0.9))
    criterion_label = nn.MSELoss()

    print(f"開始訓練 {EPOCHS} 個 Epochs...")
    print(f"Critic Iterations: {CRITIC_ITERATIONS}, Lambda GP: {LAMBDA_GP}, Beta Adv: {BETA_ADV}, Gamma L_t: {GAMMA_L_T}")

    # --- 5. 執行訓練迴圈 (WDGRL - **SDA** 邏輯) ---
    len_dataloader = min(len(source_loader), len(target_loader))
    if len_dataloader == 0: return

    for epoch in range(EPOCHS):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        avg_loss_G, avg_loss_C, avg_loss_y_s, avg_loss_y_t = 0, 0, 0, 0
        avg_wasserstein_dist = 0
        
        # *** 修改：迴圈次數由較短的 target_loader 決定 ***
        for i in range(len(target_loader)): # Iterate based on target loader
            
            ### 優化修改 ###
            # 1. 在所有迴圈之外，只載入一次
            # (處理 Source 較長的問題)
            try: X_s_batch, y_s_batch = next(source_iter)
            except StopIteration: source_iter = iter(source_loader); X_s_batch, y_s_batch = next(source_iter)
            # (Target 正常載入)
            try: X_t_batch, y_t_batch = next(target_iter) 
            except StopIteration: target_iter = iter(target_loader); X_t_batch, y_t_batch = next(target_iter)

            X_s_batch, y_s_batch = X_s_batch.to(device), y_s_batch.to(device)
            X_t_batch, y_t_batch = X_t_batch.to(device), y_t_batch.to(device)

            # --- (A) 訓練評估器 (Critic) ---
            
            ### 優化修改 ###
            # 2. 只執行一次 fe (在 no_grad 模式下)
            with torch.no_grad():
                features_s_detached = feature_extractor(X_s_batch).detach()
                features_t_detached = feature_extractor(X_t_batch).detach()

            for _ in range(CRITIC_ITERATIONS):
                ### 優化修改 ###
                # 3. 不再載入新 batch 或 執行 fe
                
                optimizer_C.zero_grad()
                
                # 4. 重複使用 'detached' 的特徵
                score_s, score_t = domain_critic(features_s_detached), domain_critic(features_t_detached)
                wasserstein_distance = score_s.mean() - score_t.mean()
                gradient_penalty = compute_gradient_penalty(domain_critic, features_s_detached.data, features_t_detached.data, device)
                loss_c = -wasserstein_distance + LAMBDA_GP * gradient_penalty
                loss_c.backward(); optimizer_C.step()
            
            # (記錄 W_dist 供顯示)
            avg_wasserstein_dist += wasserstein_distance.item()

            # --- (B) 訓練生成器 (G_f, G_y) - **SDA 修改** ---
            
            ### 優化修改 ###
            # 5. 不再載入新 batch，使用步驟 1 載入的 batch
            
            optimizer_G.zero_grad()
            
            # 6. 必須重新執行 fe (這次 *開啟* 梯度) 以便回傳
            features_s = feature_extractor(X_s_batch)
            features_t = feature_extractor(X_t_batch)
            
            # 計算 Source 標籤損失
            predictions_s = label_predictor(features_s)
            loss_y_s = criterion_label(predictions_s, y_s_batch)
            
            # *** 新增：計算 Target 標籤損失 ***
            predictions_t = label_predictor(features_t)
            loss_y_t = criterion_label(predictions_t, y_t_batch)
            
            # 計算對抗損失
            score_s_adv, score_t_adv = domain_critic(features_s), domain_critic(features_t)
            
            ### 關鍵修改 ###
            # (根據我們之前的討論，fe 的目標是最小化 L_wd，
            #  因此我們使用 L_wd = (AvgScore_S - AvgScore_T))
            # adversarial_loss = -(score_s_adv.mean() - score_t_adv.mean()) # 舊的
            adversarial_loss = score_s_adv.mean() - score_t_adv.mean() # WDGRL.pdf 論文版
            
            # *** 修改：生成器總損失加入 Target 標籤損失 ***
            loss_g = loss_y_s + (GAMMA_L_T * loss_y_t) + (BETA_ADV * adversarial_loss)
            
            loss_g.backward()
            optimizer_G.step()
            
            # 記錄損失 (loss_y_t 現在是訓練的一部分，但也記錄下來監控)
            avg_loss_G += loss_g.item(); avg_loss_C += loss_c.item()
            avg_loss_y_s += loss_y_s.item(); avg_loss_y_t += loss_y_t.item()

        # *** 修改：用 len(target_loader) 計算平均 ***
        avg_loss_G /= len(target_loader); avg_loss_C /= len(target_loader)
        avg_loss_y_s /= len(target_loader); avg_loss_y_t /= len(target_loader)
        avg_wasserstein_dist /= len(target_loader)
        
        print_interval = max(1, EPOCHS // 20) 
        if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == EPOCHS -1:
            rmse_s = np.sqrt(avg_loss_y_s); rmse_t = np.sqrt(avg_loss_y_t)
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}], Loss_G: {avg_loss_G:.4f}, Loss_C: {avg_loss_C:.4f} (W_dist: {avg_wasserstein_dist:.4f}), RMSE_S: {rmse_s:.4f}, RMSE_T: {rmse_t:.4f}")

        # 每 10 個 Epoch 儲存一次模型
        if (epoch + 1) % 10 == 0:
            fe_path = os.path.join(CHECKPOINT_DIR, f'wdgrl_mall_fe_epoch_{epoch+1}.pth')
            lp_path = os.path.join(CHECKPOINT_DIR, f'wdgrl_mall_lp_epoch_{epoch+1}.pth')
            torch.save(feature_extractor.state_dict(), fe_path)
            torch.save(label_predictor.state_dict(), lp_path)

    # --- 6. 儲存最終模型權重 (不變) ---
    torch.save(feature_extractor.state_dict(), FE_MODEL_PATH_FINAL)
    torch.save(label_predictor.state_dict(), LP_MODEL_PATH_FINAL)
    print(f"訓練完成。最終模型已儲存。Checkpoints 於 '{CHECKPOINT_DIR}/'")

if __name__ == "__main__":
    train_wdgrl_regression_sda_10_1()