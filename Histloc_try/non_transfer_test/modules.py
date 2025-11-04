import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    特徵提取器 (f_F)
    (此模型保持不變)
    """
    def __init__(self, input_dim, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        return self.network(x)

class LabelPredictor(nn.Module):
    """
    標籤預測器 (f_L)
    *** 已修改為迴歸 (Regression) ***
    """
    def __init__(self, feature_dim=64, output_dim=2):
        super(LabelPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # *** 關鍵修改：輸出維度為 2 (X, Y) ***
            # 迴歸任務的輸出層通常不需要激活函數 (如 Sigmoid 或 Softmax)
        )

    def forward(self, x):
        return self.network(x)

# 移除了 compute_differentiable_histogram 和 compute_histogram_loss
# 因為在「階段一」的訓練中不需要它們