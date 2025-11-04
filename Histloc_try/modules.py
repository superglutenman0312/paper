import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    特徵提取器 (f_F)
    """
    def __init__(self, input_dim, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            # 鑑於 Mall 資料集 AP 很多 (1033)，加大網路
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.3), # 增加 Dropout 防止過擬合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim),
            nn.Sigmoid() # 確保特徵在 0-1 之間，以便計算直方圖
        )
    
    def forward(self, x):
        return self.network(x)

class LabelPredictor(nn.Module):
    """
    標籤預測器 (f_L)
    *** 迴歸 (Regression) 版 ***
    """
    def __init__(self, feature_dim=64, output_dim=2): # 輸出 (X, Y)
        super(LabelPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # 輸出 2 維
        )

    def forward(self, x):
        return self.network(x)

# --- HistLoc 核心函式 ---

def compute_differentiable_histogram(x, bins=100, min_val=0.0, max_val=1.0):
    """
    使用高斯核 (Gaussian Kernels) 計算軟直方圖 (可微分)
    """
    bin_centers = torch.linspace(min_val, max_val, bins).to(x.device)
    bin_width = (max_val - min_val) / (bins - 1)
    
    diff = x.unsqueeze(-1) - bin_centers
    sigma = bin_width * 0.5 
    kde = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
    
    hist = kde.sum(dim=1) 
    hist = hist / hist.sum(dim=-1, keepdim=True)
    
    return hist.mean(dim=0) 

def compute_histogram_loss(features_s, features_t):
    """
    計算直方圖相關性損失 (L_D)
    L_D = 1 - Corr(H_S, H_T)
    """
    hist_s = compute_differentiable_histogram(features_s) # H_S
    hist_t = compute_differentiable_histogram(features_t) # H_T
    
    hist_s_centered = hist_s - hist_s.mean()
    hist_t_centered = hist_t - hist_t.mean()
    
    corr_numerator = (hist_s_centered * hist_t_centered).sum()
    corr_denominator = torch.sqrt((hist_s_centered ** 2).sum()) * torch.sqrt((hist_t_centered ** 2).sum())
    
    corr = corr_numerator / (corr_denominator + 1e-6)
    
    loss_d = 1.0 - corr
    return loss_d