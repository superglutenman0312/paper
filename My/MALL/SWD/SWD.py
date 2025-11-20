'''
python SWD.py --training_source_domain_data D:/paper_thesis/My/data/MTLocData/Mall/2021-11-20/wireless_training.csv `
             --training_target_domain_data D:/paper_thesis/My/data/MTLocData/Mall/2022-12-21/wireless_training.csv `
             --work_dir experiments `
             --loss_weights 0.1 10 --epoch 5 --unlabeled
python SWD.py --test --work_dir experiments `
             --loss_weights 0.000001 0.00001 --epoch 100 --unlabeled
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
        self.input_size = 1033  # 保留你的設定
        self.feature_extractor_neurons = [128, 64]

        # --- [GPU] 1. 設定 Device ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {self.device}")

        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')  # Initialize with positive infinity

    def load_train_data(self, source_data_path, target_data_path, drop_out_rate=0.0):
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        self.target_dataset = IndoorLocalizationDataset(target_data_path)

        # Drop samples from the target domain based on drop_out_rate
        if drop_out_rate > 0:
            num_samples_to_drop = int(len(self.target_dataset) * drop_out_rate)
            if drop_out_rate >= 1.0:
                num_samples_to_drop = num_samples_to_drop - 2
            drop_indices = np.random.choice(len(self.target_dataset), num_samples_to_drop, replace=False)
            self.target_dataset.data = np.delete(self.target_dataset.data, drop_indices, axis=0)

        print(f"Total Source Dataset Size: {len(self.source_dataset)}")
        print(f"Total Target Dataset Size: {len(self.target_dataset)}")

        # Split source data into training and validation sets
        source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

        # Split target data into training and validation sets
        target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
        self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
        self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

    def load_test_data(self, test_data_path):
        self.test_dataset = IndoorLocalizationDataset(test_data_path)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
        self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=298) # 保留你的設定
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)
        
        # --- [GPU] 2. 將模型搬到 GPU ---
        self.domain_adaptation_model.to(self.device)

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=self.lr)
        self.domain_criterion = nn.CrossEntropyLoss()

    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.domain_losses = [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

    def domain_invariance_loss(self, source_features, target_features):
        """
        計算 Sliced Wasserstein Distance (SWD)。
        包含自動處理 Batch Size 不一致的機制。
        """
        # --- [Fix] 處理 Batch Size 不一致的問題 ---
        batch_size_source = source_features.shape[0]
        batch_size_target = target_features.shape[0]

        if batch_size_source != batch_size_target:
            # 取兩者中較小的 batch size
            min_batch_size = min(batch_size_source, batch_size_target)
            # 截斷較大的那個，使其與較小的匹配
            source_features = source_features[:min_batch_size]
            target_features = target_features[:min_batch_size]
        # -------------------------------------------

        # 1. 設定投影數量
        num_projections = 50 
        
        # 確保投影向量在同一個裝置上 (這裡是自動抓取 input features 的 device，所以不用改)
        device = source_features.device
        
        # 取得形狀
        # 注意：這裡要用切過之後的 shape，不能用原本的
        batch_size = source_features.shape[0] 
        feature_dim = source_features.shape[1]
        
        # 2. 隨機生成投影方向矩陣
        projections = torch.randn(feature_dim, num_projections, device=device)
        projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
        
        # 3. 投影
        source_projections = torch.matmul(source_features, projections)
        target_projections = torch.matmul(target_features, projections)
        
        # 4. 排序 (Quantile Matching)
        source_sorted, _ = torch.sort(source_projections, dim=0)
        target_sorted, _ = torch.sort(target_projections, dim=0)
        
        # 5. 計算距離 (L1 Distance)
        wd_loss = torch.abs(source_sorted - target_sorted)
        
        # 6. 取平均
        return torch.mean(wd_loss)

    def train(self, num_epochs=10, unlabeled=False):
        unlabeled = unlabeled
        for epoch in range(num_epochs):
            # Training
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
            
            # Check if the current total loss is the best so far
            if self.val_total_losses[-1] < self.best_val_total_loss:
                # Save the model parameters
                print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = self.val_total_losses[-1]

    def _run_epoch(self, data_loader, training=False, unlabeled=False):
        source_correct_predictions, source_total_samples = 0, 0
        target_correct_predictions, target_total_samples = 0, 0
        total_loss_all, label_loss_all, domain_loss_all = 0.0, 0.0, 0.0
        # Create infinite iterators over datasets
        source_iter = cycle(data_loader[0])
        target_iter = cycle(data_loader[1])
        # Calculate num_batches based on the larger dataset
        num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

        for _ in range(num_batches):
            source_features, source_labels = next(source_iter)
            target_features, target_labels = next(target_iter)
            
            # --- [GPU] 3. 將資料搬到 GPU ---
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

            domain_loss = self.domain_invariance_loss(source_features, target_features)

            total_loss = self.loss_weights[0] * domain_loss + self.loss_weights[1] * label_loss
            
            total_loss_all += total_loss.item()
            label_loss_all += label_loss.item()
            domain_loss_all += domain_loss.item()
            
            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            _, source_preds = torch.max(source_labels_pred, 1)
            source_correct_predictions += (source_preds == source_labels).sum().item()
            source_total_samples += source_labels.size(0)
            source_accuracy = source_correct_predictions / source_total_samples

            _, target_preds = torch.max(target_labels_pred, 1)
            target_correct_predictions += (target_preds == target_labels).sum().item()
            target_total_samples += target_labels.size(0)
            target_accuracy = target_correct_predictions / target_total_samples
            
            acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
            
        loss_list = [total_loss_all / num_batches, label_loss_all / num_batches, domain_loss_all / num_batches]
        return loss_list, acc_list

    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        epochs_list = np.arange(0, len(self.total_losses), 1)
        label_losses_values = [loss for loss in self.label_losses]
        val_label_losses_values = [loss for loss in self.val_label_losses]

        plt.figure(figsize=(12, 8))
        
        # Subplot for Label Predictor Training Loss (Top Left)
        plt.subplot(2, 2, 1)
        plt.plot(epochs_list, label_losses_values, label='Label Loss', color='blue')
        plt.plot(epochs_list, val_label_losses_values, label='Val Label Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Label Predictor Training Loss')

        # Subplot for Training Accuracy (Top Right)
        plt.subplot(2, 2, 2)
        plt.plot(epochs_list, self.total_accuracies, label='Accuracy', color='blue')
        plt.plot(epochs_list, self.val_total_accuracies, label='Val Accuracy', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy')

        # Subplot for Domain Discriminator Training Loss (Bottom Left)
        plt.subplot(2, 2, 3)
        plt.plot(epochs_list, self.domain_losses, label='Domain Loss', color='blue')
        plt.plot(epochs_list, self.val_domain_losses, label='Val Domain Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Domain Discriminator Training Loss')

        # Remove empty subplot (Bottom Right)
        plt.subplot(2, 2, 4)
        plt.axis('off')

        # Add a title for the entire figure
        plt.suptitle('Training Curve')

        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig('loss_and_accuracy.png')
        plt.close() # 加上 close 避免佔用資源

    def load_model(self, model_path):
        if os.path.exists(model_path):
            # --- [GPU] 5. 載入模型到 GPU ---
            self.domain_adaptation_model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Error: Model file not found at {model_path}")

    def predict(self, features):
        self.domain_adaptation_model.eval()
        # --- [GPU] 4.1 預測輸入搬到 GPU ---
        features = features.to(self.device)
        with torch.no_grad():
            features, labels_pred = self.domain_adaptation_model(features)
        return labels_pred

    def generate_predictions(self, file_path, output_path):
        predictions = {'label': [], 'pred': []}
        self.load_test_data(file_path)
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                labels_pred = self.predict(test_batch)
                _, preds = torch.max(labels_pred, 1)
                predicted_labels = preds + 1  # 加 1 是为了将索引转换为 1 到 298 的标签
                label = true_label_batch + 1
                # 將預測結果保存到 predictions 中
                predictions['label'].extend(label.tolist())
                # --- [GPU] 4.2 預測輸出搬回 CPU 轉 list ---
                predictions['pred'].extend(predicted_labels.cpu().tolist())
        # 将预测结果保存为 CSV 文件
        results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
        results.to_csv(output_path, index=False)

# class HistCorrDANNModel:
#     def __init__(self, model_save_path='saved_model.pth', loss_weights=None, lr=0.001, work_dir=None):
#         if not os.path.exists(work_dir):
#             os.makedirs(work_dir)
#         os.chdir(work_dir)
#         self.batch_size = 32
#         self.loss_weights = loss_weights
#         self.lr = lr
#         self.input_size = 1033
#         self.feature_extractor_neurons = [128, 64]
        
#         # 1. 設定 Device [GPU]
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Running on device: {self.device}")

#         self._initialize_model()
#         self._initialize_optimizer()
#         self._initialize_metrics()

#         self.model_save_path = model_save_path
#         self.best_val_total_loss = float('inf')

#     def load_train_data(self, source_data_path, target_data_path, drop_out_rate=0.0):
#         self.source_dataset = IndoorLocalizationDataset(source_data_path)
#         self.target_dataset = IndoorLocalizationDataset(target_data_path)

#         if drop_out_rate > 0:
#             num_samples_to_drop = int(len(self.target_dataset) * drop_out_rate)
#             if drop_out_rate >= 1.0:
#                 num_samples_to_drop = num_samples_to_drop - 2
#             drop_indices = np.random.choice(len(self.target_dataset), num_samples_to_drop, replace=False)
#             self.target_dataset.data = np.delete(self.target_dataset.data, drop_indices, axis=0)

#         print(f"Total Source Dataset Size: {len(self.source_dataset)}")
#         print(f"Total Target Dataset Size: {len(self.target_dataset)}")

#         source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
#         self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
#         self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

#         target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
#         self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
#         self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

#     def load_test_data(self, test_data_path):
#         self.test_dataset = IndoorLocalizationDataset(test_data_path)
#         self.test_loader = DataLoader(self.test_dataset, shuffle=False)

#     def _initialize_model(self):
#         self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
#         self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=49)
#         self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)
        
#         # 2. 將模型搬到 GPU [GPU]
#         self.domain_adaptation_model.to(self.device)

#     def _initialize_optimizer(self):
#         self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=self.lr)
#         self.domain_criterion = nn.CrossEntropyLoss()

#     def _initialize_metrics(self):
#         self.total_losses, self.label_losses, self.domain_losses = [], [], []
#         self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
#         self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
#         self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

#     def domain_invariance_loss(self, source_features, target_features):
#         """
#         SWD Loss (已經包含 Batch Size 對齊)
#         注意：這裡不需要 .to(device)，因為傳進來的 features 已經在 GPU 上了
#         """
#         batch_size_source = source_features.shape[0]
#         batch_size_target = target_features.shape[0]

#         if batch_size_source != batch_size_target:
#             min_batch_size = min(batch_size_source, batch_size_target)
#             source_features = source_features[:min_batch_size]
#             target_features = target_features[:min_batch_size]

#         num_projections = 50
#         device = source_features.device # 自動獲取 GPU device
        
#         feature_dim = source_features.shape[1]
#         projections = torch.randn(feature_dim, num_projections, device=device)
#         projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
        
#         source_projections = torch.matmul(source_features, projections)
#         target_projections = torch.matmul(target_features, projections)
        
#         source_sorted, _ = torch.sort(source_projections, dim=0)
#         target_sorted, _ = torch.sort(target_projections, dim=0)
        
#         wd_loss = torch.abs(source_sorted - target_sorted)
#         return torch.mean(wd_loss)
    
#     def train(self, num_epochs=10, unlabeled=False):
#         unlabeled = unlabeled
#         for epoch in range(num_epochs):
#             loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

#             self.total_losses.append(loss_list[0])
#             self.label_losses.append(loss_list[1])
#             self.domain_losses.append(loss_list[2])
#             self.total_accuracies.append(acc_list[0])
#             self.source_accuracies.append(acc_list[1])
#             self.target_accuracies.append(acc_list[2])

#             # Validation
#             with torch.no_grad():
#                 val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

#                 self.val_total_losses.append(val_loss_list[0])
#                 self.val_label_losses.append(val_loss_list[1])
#                 self.val_domain_losses.append(val_loss_list[2])
#                 self.val_total_accuracies.append(val_acc_list[0])
#                 self.val_source_accuracies.append(val_acc_list[1])
#                 self.val_target_accuracies.append(val_acc_list[2])
            
#             print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
#             if self.val_total_losses[-1] < self.best_val_total_loss:
#                 print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
#                 self.save_model()
#                 self.best_val_total_loss = self.val_total_losses[-1]

#     def _run_epoch(self, data_loader, training=False, unlabeled=False):
#         source_correct_predictions, source_total_samples = 0, 0
#         target_correct_predictions, target_total_samples = 0, 0
        
#         source_iter = cycle(data_loader[0])
#         target_iter = cycle(data_loader[1])
#         num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

#         for _ in range(num_batches):
#             source_features, source_labels = next(source_iter)
#             target_features, target_labels = next(target_iter)
            
#             # 3. 將數據搬到 GPU [GPU]
#             source_features = source_features.to(self.device)
#             source_labels = source_labels.to(self.device)
#             target_features = target_features.to(self.device)
#             target_labels = target_labels.to(self.device)

#             source_features, source_labels_pred = self.domain_adaptation_model(source_features)
#             target_features, target_labels_pred = self.domain_adaptation_model(target_features)

#             label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
#             label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
            
#             if unlabeled:
#                 label_loss = label_loss_source
#             else:
#                 label_loss = (label_loss_source + label_loss_target) / 2

#             # 使用 SWD Loss
#             domain_loss = self.domain_invariance_loss(source_features, target_features)

#             total_loss = self.loss_weights[0] * domain_loss + self.loss_weights[1] * label_loss

#             if training:
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()

#             # 計算準確率 (在 GPU 上運算)
#             _, source_preds = torch.max(source_labels_pred, 1)
#             source_correct_predictions += (source_preds == source_labels).sum().item()
#             source_total_samples += source_labels.size(0)
#             source_accuracy = source_correct_predictions / source_total_samples

#             _, target_preds = torch.max(target_labels_pred, 1)
#             target_correct_predictions += (target_preds == target_labels).sum().item()
#             target_total_samples += target_labels.size(0)
#             target_accuracy = target_correct_predictions / target_total_samples
            
#             # 記得加 .item()
#             loss_list = [total_loss.item(), label_loss.item(), domain_loss.item()]
#             acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
            
#         return loss_list, acc_list

#     def save_model(self):
#         torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
#         print(f"Model parameters saved to {self.model_save_path}")

#     def plot_training_results(self):
#         # ... (原本的繪圖程式碼不用變，因為 loss_list 已經是 .item() 轉出的 float 了) ...
#         epochs_list = np.arange(0, len(self.total_losses), 1)
#         # ... 略 ...
#         plt.figure(figsize=(12, 8))
#         # ... 略 ...
#         plt.savefig('loss_and_accuracy.png')
#         plt.close()

#     def save_model_architecture(self, file_path='model_architecture'):
#         print(f"Model architecture saved as {file_path}")

#     def load_model(self, model_path):
#         if os.path.exists(model_path):
#             # 讀取模型到 GPU
#             self.domain_adaptation_model.load_state_dict(torch.load(model_path, map_location=self.device))
#         else:
#             print(f"Error: Model file not found at {model_path}")

#     def predict(self, features):
#         self.domain_adaptation_model.eval()
#         # 4. 預測時也要將 input 搬到 GPU [GPU]
#         features = features.to(self.device) 
#         with torch.no_grad():
#             features, labels_pred = self.domain_adaptation_model(features)
#         return labels_pred

#     def generate_predictions(self, file_path, output_path):
#         predictions = {'label': [], 'pred': []}
#         self.load_test_data(file_path)
#         with torch.no_grad():
#             for test_batch, true_label_batch in self.test_loader:
#                 # predict 裡面會處理 .to(device)
#                 labels_pred = self.predict(test_batch)
                
#                 _, preds = torch.max(labels_pred, 1)
#                 predicted_labels = preds + 1
#                 label = true_label_batch + 1
                
#                 predictions['label'].extend(label.tolist())
#                 # 因為 preds 在 GPU，所以要先 .cpu() 再 .tolist()，避免潛在問題
#                 predictions['pred'].extend(predicted_labels.cpu().tolist()) 
                
#         results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
#         results.to_csv(output_path, index=False)

# class HistCorrDANNModel:
#     def __init__(self, model_save_path='saved_model.pth', loss_weights=None, lr=0.001, work_dir=None):
#         if not os.path.exists(work_dir):
#             os.makedirs(work_dir)
#         os.chdir(work_dir)
#         self.batch_size = 32
#         self.loss_weights = loss_weights
#         self.lr = lr
#         self.input_size = 1033
#         self.feature_extractor_neurons = [128, 64]

#         self._initialize_model()
#         self._initialize_optimizer()
#         self._initialize_metrics()

#         self.model_save_path = model_save_path
#         self.best_val_total_loss = float('inf')  # Initialize with positive infinity

#     def load_train_data(self, source_data_path, target_data_path, drop_out_rate=0.0):
#         self.source_dataset = IndoorLocalizationDataset(source_data_path)
#         self.target_dataset = IndoorLocalizationDataset(target_data_path)

#         # Drop samples from the target domain based on drop_out_rate
#         if drop_out_rate > 0:
#             num_samples_to_drop = int(len(self.target_dataset) * drop_out_rate)
#             if drop_out_rate >= 1.0:
#                 num_samples_to_drop = num_samples_to_drop - 2
#             drop_indices = np.random.choice(len(self.target_dataset), num_samples_to_drop, replace=False)
#             self.target_dataset.data = np.delete(self.target_dataset.data, drop_indices, axis=0)

#         print(f"Total Source Dataset Size: {len(self.source_dataset)}")
#         print(f"Total Target Dataset Size: {len(self.target_dataset)}")

#         # Split source data into training and validation sets
#         source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
#         self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
#         self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

#         # Split target data into training and validation sets
#         target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
#         self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
#         self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

#     def load_test_data(self, test_data_path):
#         self.test_dataset = IndoorLocalizationDataset(test_data_path)
#         self.test_loader = DataLoader(self.test_dataset, shuffle=False)

#     def _initialize_model(self):
#         self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
#         self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=298)
#         self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)

#     def _initialize_optimizer(self):
#         self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=self.lr)
#         self.domain_criterion = nn.CrossEntropyLoss()

#     def _initialize_metrics(self):
#         self.total_losses, self.label_losses, self.domain_losses = [], [], []
#         self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
#         self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
#         self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

#     def domain_invariance_loss(self, source_features, target_features):
#         """
#         計算 Sliced Wasserstein Distance (SWD)。
#         包含自動處理 Batch Size 不一致的機制。
#         """
#         # --- [Fix] 處理 Batch Size 不一致的問題 ---
#         batch_size_source = source_features.shape[0]
#         batch_size_target = target_features.shape[0]

#         if batch_size_source != batch_size_target:
#             # 取兩者中較小的 batch size
#             min_batch_size = min(batch_size_source, batch_size_target)
#             # 截斷較大的那個，使其與較小的匹配
#             source_features = source_features[:min_batch_size]
#             target_features = target_features[:min_batch_size]
#         # -------------------------------------------

#         # 1. 設定投影數量
#         num_projections = 50 
        
#         # 確保投影向量在同一個裝置上
#         device = source_features.device
        
#         # 取得形狀
#         # 注意：這裡要用切過之後的 shape，不能用原本的
#         batch_size = source_features.shape[0] 
#         feature_dim = source_features.shape[1]
        
#         # 2. 隨機生成投影方向矩陣
#         projections = torch.randn(feature_dim, num_projections, device=device)
#         projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
        
#         # 3. 投影
#         source_projections = torch.matmul(source_features, projections)
#         target_projections = torch.matmul(target_features, projections)
        
#         # 4. 排序 (Quantile Matching)
#         source_sorted, _ = torch.sort(source_projections, dim=0)
#         target_sorted, _ = torch.sort(target_projections, dim=0)
        
#         # 5. 計算距離 (L1 Distance)
#         wd_loss = torch.abs(source_sorted - target_sorted)
        
#         # 6. 取平均
#         return torch.mean(wd_loss)

#     def train(self, num_epochs=10, unlabeled=False):
#         unlabeled = unlabeled
#         for epoch in range(num_epochs):
#             # Training
#             loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

#             self.total_losses.append(loss_list[0])
#             self.label_losses.append(loss_list[1])
#             self.domain_losses.append(loss_list[2])
#             self.total_accuracies.append(acc_list[0])
#             self.source_accuracies.append(acc_list[1])
#             self.target_accuracies.append(acc_list[2])

#             # Validation
#             with torch.no_grad():
#                 val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

#                 self.val_total_losses.append(val_loss_list[0])
#                 self.val_label_losses.append(val_loss_list[1])
#                 self.val_domain_losses.append(val_loss_list[2])
#                 self.val_total_accuracies.append(val_acc_list[0])
#                 self.val_source_accuracies.append(val_acc_list[1])
#                 self.val_target_accuracies.append(val_acc_list[2])
#                 # print(f'Validation Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_total_loss}, Label Loss: {val_label_loss}, Domain Loss: {val_domain_loss}, Source Accuracy: {val_source_accuracy}, Target Accuracy: {val_target_accuracy}')
            
#             # print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')
#             print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
#             # Check if the current total loss is the best so far
#             if self.val_total_losses[-1] < self.best_val_total_loss:
#                 # Save the model parameters
#                 print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
#                 self.save_model()
#                 self.best_val_total_loss = self.val_total_losses[-1]

#     def _run_epoch(self, data_loader, training=False, unlabeled=False):
#         source_correct_predictions, source_total_samples = 0, 0
#         target_correct_predictions, target_total_samples = 0, 0
#         total_loss_all, label_loss_all, domain_loss_all = 0.0, 0.0, 0.0
#         # Create infinite iterators over datasets
#         source_iter = cycle(data_loader[0])
#         target_iter = cycle(data_loader[1])
#         # Calculate num_batches based on the larger dataset
#         num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

#         for _ in range(num_batches):
#             source_features, source_labels = next(source_iter)
#             target_features, target_labels = next(target_iter)
#             source_features, source_labels_pred = self.domain_adaptation_model(source_features)
#             target_features, target_labels_pred = self.domain_adaptation_model(target_features)

#             label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
#             label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
#             if unlabeled:
#                 label_loss = label_loss_source
#             else:
#                 label_loss = (label_loss_source + label_loss_target) / 2

#             domain_loss = self.domain_invariance_loss(source_features, target_features)

#             total_loss = self.loss_weights[0] * domain_loss + self.loss_weights[1] * label_loss
            
#             total_loss_all += total_loss.item()
#             label_loss_all += label_loss.item()
#             domain_loss_all += domain_loss.item()
            
#             if training:
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()

#             _, source_preds = torch.max(source_labels_pred, 1)
#             source_correct_predictions += (source_preds == source_labels).sum().item()
#             source_total_samples += source_labels.size(0)
#             source_accuracy = source_correct_predictions / source_total_samples

#             _, target_preds = torch.max(target_labels_pred, 1)
#             target_correct_predictions += (target_preds == target_labels).sum().item()
#             target_total_samples += target_labels.size(0)
#             target_accuracy = target_correct_predictions / target_total_samples
#             # loss_list = [total_loss.item(), label_loss.item(), domain_loss]
#             acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
#         loss_list = [total_loss_all / num_batches, label_loss_all / num_batches, domain_loss_all / num_batches]
#         return loss_list, acc_list

#     def save_model(self):
#         torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
#         print(f"Model parameters saved to {self.model_save_path}")

#     def plot_training_results(self):
#         epochs_list = np.arange(0, len(self.total_losses), 1)
#         label_losses_values = [loss for loss in self.label_losses]
#         val_label_losses_values = [loss for loss in self.val_label_losses]

#         plt.figure(figsize=(12, 8))
        
#         # Subplot for Label Predictor Training Loss (Top Left)
#         plt.subplot(2, 2, 1)
#         plt.plot(epochs_list, label_losses_values, label='Label Loss', color='blue')
#         plt.plot(epochs_list, val_label_losses_values, label='Val Label Loss', color='darkorange')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.title('Label Predictor Training Loss')

#         # Subplot for Training Accuracy (Top Right)
#         plt.subplot(2, 2, 2)
#         plt.plot(epochs_list, self.total_accuracies, label='Accuracy', color='blue')
#         plt.plot(epochs_list, self.val_total_accuracies, label='Val Accuracy', color='darkorange')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.title('Training Accuracy')

#         # Subplot for Domain Discriminator Training Loss (Bottom Left)
#         plt.subplot(2, 2, 3)
#         plt.plot(epochs_list, self.domain_losses, label='Domain Loss', color='blue')
#         plt.plot(epochs_list, self.val_domain_losses, label='Val Domain Loss', color='darkorange')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.title('Domain Discriminator Training Loss')

#         # Remove empty subplot (Bottom Right)
#         plt.subplot(2, 2, 4)
#         plt.axis('off')

#         # Add a title for the entire figure
#         plt.suptitle('Training Curve')

#         plt.tight_layout()  # Adjust layout for better spacing
#         plt.savefig('loss_and_accuracy.png')

#     # def save_model_architecture(self, file_path='model_architecture'):
#     #     # Create a dummy input for visualization
#     #     dummy_input = torch.randn(1, 1033)  # Assuming input size is (batch_size, 7)

#     #     # Generate a graph of the model architecture
#     #     graph = make_dot(self.domain_adaptation_model(dummy_input), params=dict(self.domain_adaptation_model.named_parameters()))

#     #     # Save the graph as an image file
#     #     graph.render(file_path, format='png')
#     #     print(f"Model architecture saved as {file_path}")

#     def load_model(self, model_path):
#         if os.path.exists(model_path):
#             self.domain_adaptation_model.load_state_dict(torch.load(model_path))
#         else:
#             print(f"Error: Model file not found at {model_path}")

#     def predict(self, features):
#         self.domain_adaptation_model.eval()
#         with torch.no_grad():
#             features, labels_pred = self.domain_adaptation_model(features)
#         return labels_pred

#     def generate_predictions(self, file_path, output_path):
#         predictions = {'label': [], 'pred': []}
#         self.load_test_data(file_path)
#         with torch.no_grad():
#             for test_batch, true_label_batch in self.test_loader:
#                 labels_pred = self.predict(test_batch)
#                 _, preds = torch.max(labels_pred, 1)
#                 predicted_labels = preds + 1  # 加 1 是为了将索引转换为 1 到 49 的标签
#                 label = true_label_batch + 1
#                 # 將預測結果保存到 predictions 中
#                 predictions['label'].extend(label.tolist())
#                 predictions['pred'].extend(predicted_labels.tolist())
#         # 将预测结果保存为 CSV 文件
#         results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
#         results.to_csv(output_path, index=False)

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
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loss_str = f'{args.loss_weights[0]}_{args.loss_weights[1]}'
    epoch_str = f'{args.epoch}'
    status_str = 'unlabeled' if args.unlabeled else 'labeled'
    folder_name = f'{loss_str}_{epoch_str}_{status_str}'
    work_dir = os.path.join(script_dir, args.work_dir, folder_name)
    
    if args.unlabeled:
        data_drop_out_list = np.array([0.0])
    else:
        data_drop_out_list = np.array([0.9])
    
    domain1_result = []
    domain2_result = []
    domain3_result = []

    # data_drop_out_list = np.arange(0.0, 0.9, 0.95)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = HistCorrDANNModel(model_save_path=args.model_path, loss_weights=args.loss_weights, work_dir=work_dir)
        # dann_model.save_model_architecture()
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            # 訓練模型
            dann_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            dann_model.train(num_epochs=args.epoch, unlabeled=args.unlabeled)
            dann_model.plot_training_results()
        elif args.test:
            dann_model.load_model(args.model_path)
            testing_file_paths = [
                        r'D:\paper_thesis\My\data\MTLocData\Mall\2021-11-20\wireless_testing.csv',
                        r'D:\paper_thesis\My\data\MTLocData\Mall\2022-12-21\wireless_testing.csv'
                    ]
            output_paths = ['predictions/211120_results.csv', 'predictions/221221_results.csv']
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            for testing_file_path, output_path in zip(testing_file_paths, output_paths):
                dann_model.generate_predictions(testing_file_path, output_path)
                print("done")
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')
