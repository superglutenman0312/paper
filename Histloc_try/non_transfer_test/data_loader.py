import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# --- 1. 設定檔案路徑和工作表名稱 ---
TRAIN_FILE = 'Database_Scenario1.xlsx'
TEST_FILE = 'Tests_Scenario1.xlsx'
SHEET_NAME = 'WiFi' 

def get_loaders(batch_size):
    """
    從 .xlsx 檔案的特定工作表載入資料，並建立 DataLoaders
    """
    
    # --- 2. 使用 Pandas 讀取 Excel 工作表 ---
    try:
        df_train = pd.read_excel(TRAIN_FILE, sheet_name=SHEET_NAME)
        df_test = pd.read_excel(TEST_FILE, sheet_name=SHEET_NAME)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {TRAIN_FILE} 或 {TEST_FILE}")
        print("請確保 Excel 檔案與 .py 檔案在同一個資料夾中。")
        return None, None, None, 0
    except ValueError as e:
        print(f"錯誤：讀取 Excel 檔案時發生錯誤。")
        print(f"請確保工作表名稱 '{SHEET_NAME}' 存在於兩個 Excel 檔案中。")
        print(f"詳細錯誤: {e}")
        return None, None, None, 0

    # --- 3. 區分特徵 (RSSI) 和標籤 (X, Y) ---
    
    # 指定座標欄位
    label_cols = ['x', 'y']
    
    # *** 關鍵修改：明確指定 RSSI 欄位名稱 ***
    # 我們不再自動偵測，而是直接使用您提供的欄位名
    feature_cols = ['RSSI A', 'RSSI B', 'RSSI C'] 
    
    # (可選的健全性檢查) 
    # 檢查訓練資料中是否存在這些欄位
    if not all(col in df_train.columns for col in label_cols + feature_cols):
        print(f"錯誤：訓練檔案 '{TRAIN_FILE}' 中缺少必要的欄位。")
        print(f"需要 'x', 'y' 以及 'RSSI A', 'B', 'C'。")
        return None, None, None, 0
        
    # 檢查測試資料中是否存在這些欄位
    if not all(col in df_test.columns for col in label_cols + feature_cols):
        print(f"錯誤：測試檔案 '{TEST_FILE}' 中缺少必要的欄位。")
        print(f"需要 'x', 'y' 以及 'RSSI A', 'B', 'C'。")
        return None, None, None, 0
    
    num_beacons = len(feature_cols)
    print(f"成功從 '{SHEET_NAME}' 工作表讀取資料。")
    print(f"偵測到 {num_beacons} 個 AP (Beacons)。") # 這次應該會正確顯示 3
    print(f"標籤 (Labels): {label_cols}")
    print(f"特徵 (Features): {feature_cols}")

    # 取得 NumPy 陣列
    X_train_raw = df_train[feature_cols].values.astype(np.float32)
    y_train_raw = df_train[label_cols].values.astype(np.float32)
    
    X_test_raw = df_test[feature_cols].values.astype(np.float32)
    y_test_raw = df_test[label_cols].values.astype(np.float32)
    
    # --- 4. 資料標準化 (此部分邏輯不變) ---
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1)) 

    x_scaler.fit(X_train_raw)
    y_scaler.fit(y_train_raw)

    X_train_scaled = x_scaler.transform(X_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)
    
    X_test_scaled = x_scaler.transform(X_test_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)

    # --- 5. 建立 PyTorch Tensors 和 DataLoaders (此部分邏輯不變) ---
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(y_train_scaled)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled),
        torch.tensor(y_test_scaled)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_scaler, num_beacons

if __name__ == '__main__':
    # 測試
    train_loader, test_loader, y_scaler, num_beacons = get_loaders(32)
    if train_loader:
        print(f"\n成功載入資料。")
        print(f"訓練資料共 {len(train_loader.dataset)} 筆")
        print(f"測試資料共 {len(test_loader.dataset)} 筆")
        X_batch, y_batch = next(iter(train_loader))
        print(f"一批 X (Features) 的 Shape: {X_batch.shape}")
        print(f"一批 y (Labels) の Shape: {y_batch.shape}")