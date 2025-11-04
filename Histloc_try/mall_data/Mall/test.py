import h5py
import numpy as np
import os

# --- 1. 設定檔案路徑 ---
TRAIN_FILE = 'Mall_1_training.h5'
TEST_FILE  = 'Mall_1_testing.h5' 

# 建立要嘗試的路徑
TRAIN_PATHS_TO_TRY = [TRAIN_FILE, os.path.join('Mall', TRAIN_FILE)]
TEST_PATHS_TO_TRY = [TEST_FILE, os.path.join('Mall', TEST_FILE)]

def find_and_load_coords(filepaths_to_try):
    """ 從 h5 檔案安全地讀取 cdns """
    for path in filepaths_to_try:
        if not os.path.exists(path):
            continue # 檔案不存在，試下一個
        try:
            with h5py.File(path, 'r') as f:
                print(f"  > 正在讀取: {path}")
                coords = f['cdns'][:]
                return coords
        except Exception as e:
            print(f"讀取 {path} 時出錯: {e}")
            return None
    
    print(f"錯誤：在所有嘗試的路徑中都找不到檔案 (例如 {filepaths_to_try[0]})")
    return None

def verify_classification():
    print(f"正在載入訓練集座標 (RPs)...")
    train_coords = find_and_load_coords(TRAIN_PATHS_TO_TRY)
    if train_coords is None:
        return

    # --- 步驟 1: 建立 RP 座標的「唯一轉換表」---
    unique_rp_coords = np.unique(train_coords, axis=0)
    print(f"找到 {len(train_coords)} 筆訓練資料。")
    print(f"共包含 {len(unique_rp_coords)} 個唯一的 RP 座標 (Classes)。")
    
    # 為了快速查找，我們將 (x, y) 轉為字串
    rp_set = set()
    for coord in unique_rp_coords:
        rp_set.add(f"{coord[0]},{coord[1]}")

    # --- 步驟 2: 載入測試集座標 ---
    print(f"\n正在載入測試集座標 (TPs)...")
    test_coords = find_and_load_coords(TEST_PATHS_TO_TRY)
    if test_coords is None:
        return
    print(f"找到 {len(test_coords)} 筆測試資料。")

    # --- 步驟 3: 驗證 ---
    unmatched_coords = 0
    for i, coord in enumerate(test_coords):
        coord_str = f"{coord[0]},{coord[1]}"
        if coord_str not in rp_set:
            unmatched_coords += 1
            if unmatched_coords < 5: # 只顯示前幾個錯誤
                print(f"  -> 驗證失敗：測試座標 {coord} (第 {i} 筆) 不存在於訓練 RP 列表中。")

    # --- 步驟 4: 總結 ---
    if unmatched_coords == 0:
        print("\n*** 驗證成功！***")
        print("所有測試座標 (TPs) 都 100% 存在於訓練 RP 座標表中。")
        print("==> 這是一個「分類 (Classification)」問題。")
    else:
        print(f"\n*** 驗證失敗！***")
        print(f"共有 {unmatched_coords} / {len(test_coords)} 筆測試座標不存在於訓練 RP 列表中。")
        print("==> 這是一個「迴歸 (Regression)」問題。")

if __name__ == "__main__":
    verify_classification()