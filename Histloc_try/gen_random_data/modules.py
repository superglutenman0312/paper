import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    特徵提取器 (f_F)
    # 論文 3.1.2 節 提到這是一個具有兩個隱藏層的 DNN
    """
    def __init__(self, input_dim, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim), 
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid() # 確保特徵在 0-1 之間，以便計算直方圖
        )
    
    def forward(self, x):
        return self.network(x)

class LabelPredictor(nn.Module):
    """
    標籤預測器 (f_L)
    # 論文 3.1.2 節 提到這是一個簡單的神經網路，
    # 接收 X_C 作為輸入並預測 y_hat
    """
    def __init__(self, feature_dim=64, num_classes=41):
        super(LabelPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
            # nn.CrossEntropyLoss (在訓練腳本中使用) 會自動處理 softmax
        )

    def forward(self, x):
        return self.network(x)

def compute_differentiable_histogram(x, bins=100, min_val=0.0, max_val=1.0):
    """
    使用高斯核 (Gaussian Kernels) 計算軟直方圖 (可微分)
    這是為了實現論文中 提到的計算 H_S 和 H_T
    x: 輸入張量 (batch_size, feature_dim)
    """
    # 論文提到分區為 100 個 bin，範圍 0 到 1
    bin_centers = torch.linspace(min_val, max_val, bins).to(x.device)
    bin_width = (max_val - min_val) / (bins - 1)
    
    diff = x.unsqueeze(-1) - bin_centers
    sigma = bin_width * 0.5 # 高斯核的標準差
    kde = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
    
    hist = kde.sum(dim=1) # 沿著特徵維度相加
    hist = hist / hist.sum(dim=-1, keepdim=True)
    
    return hist.mean(dim=0) # 返回批次的平均直方圖

def compute_histogram_loss(features_s, features_t):
    """
    計算直方圖相關性損失 (L_D)
    L_D = 1 - Corr(H_S, H_T)
    # 損失函數定義來自論文
    # Corr 的計算方式 (皮爾遜相關係數) 來自論文
    """
    # 1. 計算直方圖
    hist_s = compute_differentiable_histogram(features_s) # H_S
    hist_t = compute_differentiable_histogram(features_t) # H_T
    
    # 2. 計算皮爾遜相關係數 (Corr)
    hist_s_centered = hist_s - hist_s.mean()
    hist_t_centered = hist_t - hist_t.mean()
    
    corr_numerator = (hist_s_centered * hist_t_centered).sum()
    corr_denominator = torch.sqrt((hist_s_centered ** 2).sum()) * torch.sqrt((hist_t_centered ** 2).sum())
    
    # 避免除以零
    corr = corr_numerator / (corr_denominator + 1e-6)
    
    # 3. 計算損失 L_D
    loss_d = 1.0 - corr
    return loss_d