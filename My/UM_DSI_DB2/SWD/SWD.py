r'''
# time variation 1
python SWD.py --training_source_domain_data D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20191009\source_train.csv `
             --training_target_domain_data D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20191009\target_train.csv `
             --work_dir time_variation_1 `
             --loss_weights 1 10 --epoch 100 --random_seed 70 --unlabeled 
python SWD.py --test --work_dir time_variation_1 `
             --loss_weights 1 10 --epoch 100 --random_seed 70 --unlabeled
             
# time variation 2
python SWD.py --training_source_domain_data D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20200219\source_train.csv `
             --training_target_domain_data D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20200219\target_train.csv `
             --work_dir test `
             --loss_weights 1 10 --epoch 100 --random_seed 70 --unlabeled
python SWD.py --test --work_dir test `
             --loss_weights 1 10 --epoch 100 --random_seed 70 --unlabeled
'''
import torch
import torch.nn as nn
from itertools import cycle
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from torchviz import make_dot
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import argparse
import os
import sys
import random

def set_seed(seed=42):
    """
    固定所有隨機種子，確保實驗可重現。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 為了絕對的一致性，犧牲一點效能（選擇性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

class LabelPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x

class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, label_predictor):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        features = self.feature_extractor(x)
        labels = self.label_predictor(features)
        return features, labels

class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class HistCorrDANNModel:
    def __init__(self, model_save_path='saved_model.pth', loss_weights=None, lr=0.001, work_dir=None):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.batch_size = 32
        self.loss_weights = loss_weights
        self.lr = lr
        self.input_size = 168
        self.feature_extractor_neurons = [128, 64]
        
        # 設定投影數量 (建議至少大於特徵維度 64，這裡設 128 以獲得更穩定的梯度)
        self.num_projections = 1024
        self.feature_dim = self.feature_extractor_neurons[1] # 64

        # 1. 設定 Device [GPU]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {self.device}")
        
        self._initialize_model()
        self._initialize_projections()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')
    
    def load_train_data(self, source_data_path, target_data_path, drop_out_rate=0.0):
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        self.target_dataset = IndoorLocalizationDataset(target_data_path)

        if drop_out_rate > 0:
            num_samples_to_drop = int(len(self.target_dataset) * drop_out_rate)
            if drop_out_rate >= 1.0:
                num_samples_to_drop = num_samples_to_drop - 2
            drop_indices = np.random.choice(len(self.target_dataset), num_samples_to_drop, replace=False)
            self.target_dataset.data = np.delete(self.target_dataset.data, drop_indices, axis=0)

        print(f"Total Source Dataset Size: {len(self.source_dataset)}")
        print(f"Total Target Dataset Size: {len(self.target_dataset)}")

        source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

        target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
        self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
        self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

    def load_test_data(self, test_data_path):
        self.test_dataset = IndoorLocalizationDataset(test_data_path)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
        self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=49)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)
        
        # 2. 將模型搬到 GPU [GPU]
        self.domain_adaptation_model.to(self.device)

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=self.lr)
        self.domain_criterion = nn.CrossEntropyLoss()

    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.domain_losses = [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

    def _get_hadamard_matrix(self, n):
        """
        遞迴生成 n x n 的 Hadamard 矩陣 (n 必須是 2 的冪次)
        """
        if n == 1:
            return torch.tensor([[1.0]])
        
        h_half = self._get_hadamard_matrix(n // 2)
        # 構造: [[H, H], [H, -H]]
        row1 = torch.cat((h_half, h_half), dim=1)
        row2 = torch.cat((h_half, -h_half), dim=1)
        h = torch.cat((row1, row2), dim=0)
        return h
    
    def _initialize_projections(self):
        """
        [修正版] 支援任意 num_projections 數量的 Hadamard 互補基底生成。
        迴圈執行，直到湊滿指定的數量。
        """
        projections_list = []
        current_count = 0
        
        # 預先計算 Hadamard 矩陣 (只算一次)
        if (self.feature_dim & (self.feature_dim - 1) == 0) and self.feature_dim != 0:
            h_mat = self._get_hadamard_matrix(self.feature_dim).to(self.device)
            h_mat = h_mat / torch.sqrt(torch.tensor(float(self.feature_dim)))
            use_hadamard = True
        else:
            use_hadamard = False
            print("Feature dim not power of 2, Hadamard disabled.")

        while current_count < self.num_projections:
            # 1. 生成隨機正交基底 (Base)
            rand_mat = torch.randn(self.feature_dim, self.feature_dim, device=self.device)
            q_base, _ = torch.linalg.qr(rand_mat)
            projections_list.append(q_base)
            current_count += self.feature_dim
            
            # 如果已經湊滿了，就提早結束 (避免多算)
            if current_count >= self.num_projections:
                break

            # 2. 生成互補基底 (Rotated by Hadamard)
            if use_hadamard:
                q_rotated = torch.matmul(q_base, h_mat)
                projections_list.append(q_rotated)
                current_count += self.feature_dim
            else:
                # 如果不能用 Hadamard，就再生成一組隨機的
                rand_mat2 = torch.randn(self.feature_dim, self.feature_dim, device=self.device)
                q2, _ = torch.linalg.qr(rand_mat2)
                projections_list.append(q2)
                current_count += self.feature_dim

        # 3. 合併並截斷到精確數量
        self.projections = torch.cat(projections_list, dim=1)
        self.projections = self.projections[:, :self.num_projections]
        
        # 確保不需要梯度
        self.projections = self.projections.detach()
        print(f"Initialized extended orthogonal projections shape: {self.projections.shape}")

    # def _initialize_projections(self):
    #     """
    #     [修改點] 生成固定的正交投影矩陣，並放入 GPU。
    #     這相當於在高維空間中建立了一組固定的「座標軸」，
    #     避免了每次隨機生成的變異，也確保方向不重複。
    #     """
    #     # 為了確保覆蓋率，我們生成多組正交基底來填滿 num_projections
    #     # 每次生成一個 64x64 的正交矩陣，直到湊滿 128 個向量
    #     projections_list = []
    #     remaining = self.num_projections
        
    #     while remaining > 0:
    #         # 生成隨機矩陣
    #         rand_mat = torch.randn(self.feature_dim, self.feature_dim)
    #         # QR 分解獲取正交矩陣 (Q 是正交的)
    #         q, _ = torch.linalg.qr(rand_mat)
            
    #         # 取出需要的數量
    #         num_to_take = min(self.feature_dim, remaining)
    #         projections_list.append(q[:, :num_to_take])
    #         remaining -= num_to_take
            
    #     # 拼接並確保歸一化 (雖然 QR 分解後的 Q 已經是歸一化的，但保險起見)
    #     self.projections = torch.cat(projections_list, dim=1)
    #     # 將固定的投影矩陣搬到 GPU，這是一個 Parameter 但不需要被 Optimizer 更新 (requires_grad=False)
    #     self.projections = self.projections.to(self.device).detach()
    #     print(f"Initialized fixed orthogonal projections shape: {self.projections.shape}")

    def domain_invariance_loss(self, source_features, target_features):
        """
        SWD Loss (使用預先生成的固定正交投影)
        """
        batch_size_source = source_features.shape[0]
        batch_size_target = target_features.shape[0]

        if batch_size_source != batch_size_target:
            min_batch_size = min(batch_size_source, batch_size_target)
            source_features = source_features[:min_batch_size]
            target_features = target_features[:min_batch_size]

        # [修改點] 不再這裡隨機生成，直接使用 self.projections
        # self.projections 已經在 GPU 上了，不需要再 .to(device)
        
        # source_features: (Batch, 64) x projections: (64, 128) -> (Batch, 128)
        source_projections = torch.matmul(source_features, self.projections)
        target_projections = torch.matmul(target_features, self.projections)
        
        # 排序 (Sorting is the core of Wasserstein distance in 1D)
        source_sorted, _ = torch.sort(source_projections, dim=0)
        target_sorted, _ = torch.sort(target_projections, dim=0)
        
        # 計算距離
        wd_loss = torch.abs(source_sorted - target_sorted)
        
        # 這裡我們對所有投影方向取平均，代表對高維分佈差異的估計
        return torch.mean(wd_loss)

    # def domain_invariance_loss(self, source_features, target_features):
    #     """
    #     SWD Loss (已經包含 Batch Size 對齊)
    #     注意：這裡不需要 .to(device)，因為傳進來的 features 已經在 GPU 上了
    #     """
    #     batch_size_source = source_features.shape[0]
    #     batch_size_target = target_features.shape[0]

    #     if batch_size_source != batch_size_target:
    #         min_batch_size = min(batch_size_source, batch_size_target)
    #         source_features = source_features[:min_batch_size]
    #         target_features = target_features[:min_batch_size]

    #     num_projections = 50
    #     device = source_features.device # 自動獲取 GPU device
        
    #     feature_dim = source_features.shape[1]
    #     projections = torch.randn(feature_dim, num_projections, device=device)
    #     projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
        
    #     source_projections = torch.matmul(source_features, projections)
    #     target_projections = torch.matmul(target_features, projections)
        
    #     source_sorted, _ = torch.sort(source_projections, dim=0)
    #     target_sorted, _ = torch.sort(target_projections, dim=0)
        
    #     wd_loss = torch.abs(source_sorted - target_sorted)
    #     return torch.mean(wd_loss)
    
    def train(self, num_epochs=10, unlabeled=False):
        unlabeled = unlabeled
        for epoch in range(num_epochs):
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.domain_losses.append(loss_list[2])
            self.total_accuracies.append(acc_list[0])
            self.source_accuracies.append(acc_list[1])
            self.target_accuracies.append(acc_list[2])

            # Validation
            with torch.no_grad():
                val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

                self.val_total_losses.append(val_loss_list[0])
                self.val_label_losses.append(val_loss_list[1])
                self.val_domain_losses.append(val_loss_list[2])
                self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
            
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
            if self.val_total_losses[-1] < self.best_val_total_loss:
                print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = self.val_total_losses[-1]

    def _run_epoch(self, data_loader, training=False, unlabeled=False):
        source_correct_predictions, source_total_samples = 0, 0
        target_correct_predictions, target_total_samples = 0, 0
        
        source_iter = cycle(data_loader[0])
        target_iter = cycle(data_loader[1])
        num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

        for _ in range(num_batches):
            source_features, source_labels = next(source_iter)
            target_features, target_labels = next(target_iter)
            
            # 3. 將數據搬到 GPU [GPU]
            source_features = source_features.to(self.device)
            source_labels = source_labels.to(self.device)
            target_features = target_features.to(self.device)
            target_labels = target_labels.to(self.device)

            source_features, source_labels_pred = self.domain_adaptation_model(source_features)
            target_features, target_labels_pred = self.domain_adaptation_model(target_features)

            label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
            label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
            
            if unlabeled:
                label_loss = label_loss_source
            else:
                label_loss = (label_loss_source + label_loss_target) / 2

            # 使用 SWD Loss
            domain_loss = self.domain_invariance_loss(source_features, target_features)

            total_loss = self.loss_weights[0] * domain_loss + self.loss_weights[1] * label_loss

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # 計算準確率 (在 GPU 上運算)
            _, source_preds = torch.max(source_labels_pred, 1)
            source_correct_predictions += (source_preds == source_labels).sum().item()
            source_total_samples += source_labels.size(0)
            source_accuracy = source_correct_predictions / source_total_samples

            _, target_preds = torch.max(target_labels_pred, 1)
            target_correct_predictions += (target_preds == target_labels).sum().item()
            target_total_samples += target_labels.size(0)
            target_accuracy = target_correct_predictions / target_total_samples
            
            # 記得加 .item()
            loss_list = [total_loss.item(), label_loss.item(), domain_loss.item()]
            acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
            
        return loss_list, acc_list

    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        # ... (原本的繪圖程式碼不用變，因為 loss_list 已經是 .item() 轉出的 float 了) ...
        epochs_list = np.arange(0, len(self.total_losses), 1)
        # ... 略 ...
        plt.figure(figsize=(12, 8))
        # ... 略 ...
        plt.savefig('loss_and_accuracy.png')
        plt.close()

    def save_model_architecture(self, file_path='model_architecture'):
        print(f"Model architecture saved as {file_path}")

    def load_model(self, model_path):
        if os.path.exists(model_path):
            # 讀取模型到 GPU
            self.domain_adaptation_model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Error: Model file not found at {model_path}")

    def predict(self, features):
        self.domain_adaptation_model.eval()
        # 4. 預測時也要將 input 搬到 GPU [GPU]
        features = features.to(self.device) 
        with torch.no_grad():
            features, labels_pred = self.domain_adaptation_model(features)
        return labels_pred

    def generate_predictions(self, file_path, output_path):
        predictions = {'label': [], 'pred': []}
        self.load_test_data(file_path)
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                # predict 裡面會處理 .to(device)
                labels_pred = self.predict(test_batch)
                
                _, preds = torch.max(labels_pred, 1)
                predicted_labels = preds + 1
                label = true_label_batch + 1
                
                predictions['label'].extend(label.tolist())
                # 因為 preds 在 GPU，所以要先 .cpu() 再 .tolist()，避免潛在問題
                predictions['pred'].extend(predicted_labels.cpu().tolist()) 
                
        results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
        results.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--test', action='store_true' , help='for test')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DANN_CORR', help='create new directory to save result')
    parser.add_argument('--loss_weights', type=float, nargs=2, default=[0.1, 10.0], help='loss weights for domain and label predictors')
    parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--unlabeled', action='store_true', help='use unlabeled data from target domain during training')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()
    
    seed = args.random_seed
    set_seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loss_str = f'{args.loss_weights[0]}_{args.loss_weights[1]}'
    epoch_str = f'{args.epoch}'
    status_str = 'unlabeled' if args.unlabeled else 'labeled'
    folder_name = f'{loss_str}_{epoch_str}_{status_str}'
    work_dir = os.path.join(script_dir, args.work_dir, f'random_seed_{seed}', folder_name)
    
    if args.unlabeled:
        data_drop_out_list = np.array([0.0])
    else:
        data_drop_out_list = np.array([0.9])
    
    domain1_result = []
    domain2_result = []
    domain3_result = []

    # data_drop_out_list = np.arange(0.9, 0.95, 0.1)

    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = HistCorrDANNModel(model_save_path=args.model_path, loss_weights=args.loss_weights, work_dir=work_dir)
        dann_model.save_model_architecture()
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            # 訓練模型
            dann_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            dann_model.train(num_epochs=args.epoch, unlabeled=args.unlabeled)
            dann_model.plot_training_results()
        elif args.test:
            dann_model.load_model(args.model_path)
            testing_file_paths = [
                        r'D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20200219\source_test.csv',
                        r'D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20191009\target_test.csv',
                        r'D:\paper_thesis\My\data\UM_DSI_DB_v1.0.0_lite\data\processed_data\20190611_20200219\target_test.csv',
                    ]
            output_paths = ['predictions/190611_results.csv', 'predictions/191009_results.csv', 'predictions/200219_results.csv']
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            for testing_file_path, output_path in zip(testing_file_paths, output_paths):
                dann_model.generate_predictions(testing_file_path, output_path)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')
