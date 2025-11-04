import h5py
import numpy as np

# 請將 'file_path' 換成您下載的 HDF5 檔案的實際路徑
file_path = 'Mall_1_training.h5'

def check(file_path):
    try:
        # 1. 使用 'r' (唯讀模式) 開啟檔案
        with h5py.File(file_path, 'r') as f:
            
            # 2. 查看檔案中所有的 "Keys" (頂層資料集)
            # 根據您的描述，這應該會印出 ['bssids', 'cdns', 'rssis', 'RecordsNums']
            print(f"檔案 '{file_path}' 中的 Keys: {list(f.keys())}")
            
            print("-" * 30)

            # 3. 讀取 'rssis' 資料 (RSSI 訊號強度矩陣)
            if 'rssis' in f:
                # f['rssis'] 是一個 HDF5 Dataset 物件
                # 使用 [:] 將資料完整讀取到記憶體中 (會變成一個 NumPy Array)
                rssi_data = f['rssis'][:]
                print(f"--- 讀取 'rssis' (訊號強度) ---")
                print(f"資料形狀 (Shape): {rssi_data.shape}")
                print(f"資料類型 (Dtype): {rssi_data.dtype}")
                print("前 2 筆 RSSI 資料 (範例):")
                print(rssi_data[:2,400:420]) # 印出前 2 筆指紋
                print("後 2 筆 RSSI 資料 (範例):")
                print(rssi_data[-2:,400:420]) # 印出前 2 筆指紋

            print("-" * 30)

            # 4. 讀取 'cdns' 資料 (座標)
            if 'cdns' in f:
                coord_data = f['cdns'][:]
                print(f"--- 讀取 'cdns' (座標) ---")
                print(f"資料形狀 (Shape): {coord_data.shape}")
                print("前 5 筆座標 (範例):")
                print(coord_data[:5]) # 印出前 5 筆座標
                print("後 5 筆座標 (範例):")
                print(coord_data[-5:]) # 印出前 5 筆座標

            print("-" * 30)

            # 5. 讀取 'bssids' 資料 (AP 的 MAC 位址)
            if 'bssids' in f:
                bssid_data = f['bssids'][:]
                print(f"--- 讀取 'bssids' (AP 列表) ---")
                print(f"資料形狀 (Shape): {bssid_data.shape}")
                print("前 5 筆 BSSID (範例):")
                # BSSID 可能是字串，需要解碼 (decode)
                if bssid_data.dtype == 'object': # H5py 常將字串存為 'object'
                    print([s.decode('utf-8') for s in bssid_data[:5]])
                else:
                    print(bssid_data[:5])
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認路徑是否正確。")
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        
check(file_path)
check('Mall_7_training.h5')
check('Mall_7_testing.h5')