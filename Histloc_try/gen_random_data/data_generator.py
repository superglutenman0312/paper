import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def get_data_loaders(num_beacons, num_classes, n_source, n_target, batch_size):
    """
    生成模擬資料並返回 DataLoaders 和測試資料
    """
    
    # --- 步驟 1: 生成隨機資料 (請替換為您的真實資料) ---
    # 模擬 Source Domain 資料
    X_source_rssi = np.random.rand(n_source, num_beacons) * -80
    y_source_labels = np.random.randint(0, num_classes, size=n_source)
    
    # 模擬 Target Domain 資料 (不同的 RSSI 均值，模擬環境變化)
    # [cite_start]模擬情境 4.2.1 [cite: 460]，目標域資料量僅為源域的 1/10
    X_target_rssi = np.random.rand(n_target, num_beacons) * -75 
    y_target_labels = np.random.randint(0, num_classes, size=n_target)

    # --- 步驟 2: 資料預處理 (Min-Max 標準化) ---
    # [cite_start]論文 3.1.1 節 [cite: 253-255] 提到使用 Min-Max
    scaler_s = MinMaxScaler(feature_range=(0, 1))
    X_s_normalized = scaler_s.fit_transform(X_source_rssi)

    scaler_t = MinMaxScaler(feature_range=(0, 1))
    X_t_normalized = scaler_t.fit_transform(X_target_rssi)

    # --- 步驟 3: 轉換為 PyTorch Tensors ---
    X_s = torch.tensor(X_s_normalized, dtype=torch.float32)
    y_s = torch.tensor(y_source_labels, dtype=torch.long)
    X_t = torch.tensor(X_t_normalized, dtype=torch.float32)
    y_t = torch.tensor(y_target_labels, dtype=torch.long)

    # --- 步驟 4: 創建 DataLoader ---
    source_dataset = TensorDataset(X_s, y_s)
    target_dataset = TensorDataset(X_t, y_t)

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 返回 Online Phase 預測時需要的資料
    # (X_target_rssi_raw, y_target_labels, scaler_for_target)
    test_data_package = (X_target_rssi, y_target_labels, scaler_t)

    return source_loader, target_loader, test_data_package