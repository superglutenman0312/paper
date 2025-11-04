import numpy as np
import torch
import h5py
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # 用於抽取子集

# --- 檔案路徑設定 (不變) ---
SOURCE_TRAIN_FILE = os.path.join('.', 'Mall_1_training.h5')
TARGET_TRAIN_FILE = os.path.join('.', 'Mall_7_training.h5') 
TARGET_TEST_FILE  = os.path.join('.', 'Mall_7_testing.h5')
# TARGET_TEST_FILE  = os.path.join('.', 'Mall_1_training.h5')
# TARGET_TEST_FILE  = os.path.join('.', 'Mall_7_training.h5')  
MISSING_RSSI_VALUE = -100.0

def load_h5_data(filepath):
    # ... (此函數不變)
    try:
        with h5py.File(filepath, 'r') as f:
            rssis = f['rssis'][:] 
            coords = f['cdns'][:]
            bssids_raw = f['bssids'][:]
            bssids = [b.decode('utf-8') for b in bssids_raw]
            # rssis[rssis == 100] = MISSING_RSSI_VALUE
            return rssis.astype(np.float32), coords.astype(np.float32), bssids
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {filepath}")
        return None, None, None
    except Exception as e:
        print(f"讀取 {filepath} 時出錯: {e}")
        return None, None, None

# *** 修改：增加 target_train_ratio 參數 ***
def get_mall_loaders(batch_size, target_train_ratio=1.0, random_seed=42):
    """
    準備 Mall 資料集，執行 BSSID 對齊，並可選擇 Target 訓練資料的比例。
    target_train_ratio: 要使用的 Target 訓練資料的比例 (0.0 到 1.0)。
    random_seed: 用於抽取 Target 子集時的隨機種子，確保可重複性。
    """
    
    # --- 1. 載入所有資料 ---
    X_s_raw, y_s_raw, bssids_s = load_h5_data(SOURCE_TRAIN_FILE)
    X_t_train_full_raw, y_t_train_full_raw, bssids_t_train = load_h5_data(TARGET_TRAIN_FILE)
    X_t_test_raw, y_t_test_raw, bssids_t_test = load_h5_data(TARGET_TEST_FILE)

    if X_s_raw is None or X_t_train_full_raw is None or X_t_test_raw is None:
        return None, None, None, None, 0, 0 # 返回額外的 None 代表 source_size

    print(f"原始 AP 數量 -> Source: {len(bssids_s)}, Target-Train: {len(bssids_t_train)}, Target-Test: {len(bssids_t_test)}")

    # --- 2. BSSID 對齊 (不變) ---
    print("開始 BSSID 對齊...")
    set_s, set_t_train, set_t_test = set(bssids_s), set(bssids_t_train), set(bssids_t_test)
    all_bssids_set = set_s.union(set_t_train).union(set_t_test)
    all_bssids_list = sorted(list(all_bssids_set))
    num_beacons_aligned = len(all_bssids_list)
    bssid_to_index = {bssid: idx for idx, bssid in enumerate(all_bssids_list)}

    X_s_aligned = np.full((X_s_raw.shape[0], num_beacons_aligned), MISSING_RSSI_VALUE, dtype=np.float32)
    X_t_train_full_aligned = np.full((X_t_train_full_raw.shape[0], num_beacons_aligned), MISSING_RSSI_VALUE, dtype=np.float32)
    X_t_test_aligned = np.full((X_t_test_raw.shape[0], num_beacons_aligned), MISSING_RSSI_VALUE, dtype=np.float32)

    source_indices = [bssid_to_index[b] for b in bssids_s]
    target_train_indices = [bssid_to_index[b] for b in bssids_t_train]
    target_test_indices = [bssid_to_index[b] for b in bssids_t_test]

    X_s_aligned[:, source_indices] = X_s_raw
    X_t_train_full_aligned[:, target_train_indices] = X_t_train_full_raw
    X_t_test_aligned[:, target_test_indices] = X_t_test_raw
    print(f"BSSID 聯集 (總 AP 數): {num_beacons_aligned}")
    print("BSSID 對齊完成。")

    # --- 3. *** 新增：抽取 Target 訓練資料子集 *** ---
    if target_train_ratio < 1.0 and target_train_ratio > 0.0:
        print(f"從 Target 訓練資料中抽取 {target_train_ratio*100:.1f}% 的子集...")
        # 使用 train_test_split 來隨機抽取，並確保標籤分佈大致均衡（如果可能）
        # 我們實際上只需要 "train" 部分，所以 test_size 設為 1-ratio
        X_t_train_aligned, _, y_t_train_raw, _ = train_test_split(
            X_t_train_full_aligned, 
            y_t_train_full_raw, 
            train_size=target_train_ratio, 
            random_state=random_seed,
            # stratify=y_t_train_full_raw # 迴歸任務通常不進行分層抽樣
        )
        print(f"抽取完成，Target 訓練子集大小: {X_t_train_aligned.shape[0]}")
    elif target_train_ratio == 1.0:
         X_t_train_aligned = X_t_train_full_aligned
         y_t_train_raw = y_t_train_full_raw
         print("使用全部 Target 訓練資料。")
    else:
        print(f"錯誤：target_train_ratio ({target_train_ratio}) 必須介於 (0, 1]。")
        return None, None, None, None, 0, 0

    # --- 4. 資料標準化 (現在使用 X_t_train_aligned) ---
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1)) 

    # Fit 只能用訓練資料 (Source + Target Train 子集)
    x_scaler.fit(np.vstack((X_s_aligned, X_t_train_aligned)))
    y_scaler.fit(np.vstack((y_s_raw, y_t_train_raw))) # <-- 使用 y_t_train_raw 子集

    # Transform 所有資料
    X_s_scaled = x_scaler.transform(X_s_aligned)
    y_s_scaled = y_scaler.transform(y_s_raw)
    
    X_t_train_scaled = x_scaler.transform(X_t_train_aligned) # <-- 使用對齊後的子集
    y_t_train_scaled = y_scaler.transform(y_t_train_raw) # <-- 使用標籤子集

    X_t_test_scaled = x_scaler.transform(X_t_test_aligned) # Test 集保持不變
    y_t_test_scaled = y_scaler.transform(y_t_test_raw) # Test 集保持不變

    # --- 5. 建立 Tensors 和 DataLoaders ---
    ds_source = TensorDataset(torch.tensor(X_s_scaled), torch.tensor(y_s_scaled))
    ds_target_train = TensorDataset(torch.tensor(X_t_train_scaled), torch.tensor(y_t_train_scaled)) # <-- 使用子集
    ds_target_test = TensorDataset(torch.tensor(X_t_test_scaled), torch.tensor(y_t_test_scaled))

    source_loader = DataLoader(ds_source, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(ds_target_train, batch_size=batch_size, shuffle=True, drop_last=True) # <-- 使用子集
    test_loader = DataLoader(ds_target_test, batch_size=batch_size, shuffle=False)
    
    # 返回 source data size 以便計算 ratio
    source_size = X_s_aligned.shape[0] 
    
    return source_loader, target_loader, test_loader, y_scaler, num_beacons_aligned, source_size

if __name__ == '__main__':
    # 測試抽取 10%
    loaders = get_mall_loaders(32, target_train_ratio=0.1) 
    if loaders[0]:
        print("\n資料載入成功。")
        (s_loader, t_loader, test_l, _, num_aps, src_size) = loaders
        print(f"Source Loader 大小 (批次數): {len(s_loader)}, 總樣本數: {src_size}")
        print(f"Target Train Loader 大小 (批次數): {len(t_loader)}, 總樣本數: {len(t_loader.dataset)}") # 檢查樣本數是否約為 Source 的 1/10
        print(f"Test Loader 大小 (批次數): {len(test_l)}")