import torch
import torch.nn as nn
from torch.autograd import Function

# --- 1. 梯度反轉層 (GRL) ---
# 這是 DANN 論文的核心
class GradientReversalFunc(Function):
    """
    PyTorch 自定義函式，用於實現 GRL。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向傳播：GRL 是一個恆等函數 (Identity)。
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向傳播：GRL 將梯度乘以 -alpha。
        這會反轉 G_f 關於 L_d 的梯度，實現「最大化」L_d 的效果。
        """
        # grad_output.neg() * ctx.alpha
        output = grad_output.neg() * ctx.alpha
        return output, None # 梯度只傳回給 x，不傳回給 alpha

class GradientReversalLayer(nn.Module):
    """
    GRL 的 nn.Module 包裝
    """
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x, alpha=1.0):
        """
        GRL 層，alpha 是反轉的強度
        """
        return GradientReversalFunc.apply(x, alpha)


# --- 2. 特徵提取器 (G_f) ---
class FeatureExtractor(nn.Module):
    """
    特徵提取器 (G_f)
    (與 HistLoc 的 FeatureExtractor 相同)
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
            nn.Sigmoid() # DANN 論文的特徵層通常也用 Tanh 或 Sigmoid
        )
    
    def forward(self, x):
        return self.network(x)


# --- 3. 標籤預測器 (G_y) ---
class LabelPredictor(nn.Module):
    """
    標籤預測器 (G_y)
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


# --- 4. 域判別器 (G_d) ---
class DomainDiscriminator(nn.Module):
    """
    域判別器 (G_d)
    這是一個二元分類器，用於判斷特徵來自 Source (0) 還是 Target (1)
    """
    def __init__(self, feature_dim, hidden_dim=32):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 2 個類別: 0 (Source) vs 1 (Target)
        )
    
    def forward(self, x):
        return self.network(x)