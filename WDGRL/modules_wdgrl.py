import torch
import torch.nn as nn

# --- 1. 特徵提取器 (G_f) ---
class FeatureExtractor(nn.Module):
    """
    特徵提取器 (G_f)
    (與 DANN / HistLoc 的 FeatureExtractor 相同)
    """
    def __init__(self, input_dim, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim),
            nn.Sigmoid() # WDGRL 通常也需要特徵在一定範圍內
        )
    
    def forward(self, x):
        return self.network(x)


# --- 2. 標籤預測器 (G_y) ---
class LabelPredictor(nn.Module):
    """
    標籤預測器 (G_y)
    *** 迴歸 (Regression) 版 ***
    """
    # def __init__(self, feature_dim=64, output_dim=2): # 輸出 (X, Y)
    #     super(LabelPredictor, self).__init__()
    #     self.network = nn.Sequential(
    #         nn.Linear(feature_dim, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, output_dim) # 輸出 2 維
    #     )

    def __init__(self, feature_dim=64, output_dim=2): # 輸出 (X, Y)
        super(LabelPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, output_dim) # 輸出 2 維
        )

    def forward(self, x):
        return self.network(x)


# --- 3. 域評估器 (Domain Critic, f) ---
class DomainCritic(nn.Module):
    """
    域評估器 (Domain Critic, f)
    這個網路輸出一個純量分數，而不是分類機率
    """
    def __init__(self, feature_dim, hidden_dim=32):
        super(DomainCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # *** 關鍵：輸出 1 維 (純量分數) ***
        )
    
    def forward(self, x):
        return self.network(x)

# --- 4. 梯度懲罰 (Gradient Penalty) ---
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    計算梯度懲罰 (Gradient Penalty)
    """
    # 隨機插值
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand(real_samples.size())
    
    # 創建插值樣本
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 計算評估器在插值樣本上的輸出
    critic_interpolates = critic(interpolates)
    
    # 計算梯度
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 計算梯度的 L2 範數
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty