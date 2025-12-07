r'''
# time variation
python DNN.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\220318\GalaxyA51\wireless_training.csv `
                      --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\231116\GalaxyA51\wireless_training.csv `
                      --work_dir time_variation `
                      --random_seed 42 --unlabeled 
python DNN.py --test --work_dir time_variation `
                      --random_seed 42 --unlabeled
# spatial variation
python DNN.py --training_source_domain_data D:\paper_thesis\Histloc_real\Experiment\data\231116\GalaxyA51\wireless_training.csv `
                      --training_target_domain_data D:\paper_thesis\Histloc_real\Experiment\data\231117\GalaxyA51\wireless_training.csv `
                      --work_dir spatial_variation `
                      --random_seed 42 --unlabeled 
python DNN.py --test --work_dir spatial_variation `
                      --random_seed 42 --unlabeled
'''

import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
import os
import random
import sys

def set_seed(seed=42):
    """
    固定所有隨機種子，確保實驗可重現。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x

class SimpleDNN(nn.Module):
    def __init__(self, feature_extractor, label_predictor):
        super(SimpleDNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.label_predictor(features)
        return logits

class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class HistCorrDNNModel:
    def __init__(self, model_save_path='saved_dnn_model.pth', lr=0.001, work_dir=None):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.batch_size = 32
        self.lr = lr
        self.input_size = 7
        self.feature_extractor_neurons = [64, 48]
        
        # 設定 Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {self.device}")

        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_loss = float('inf')

    def load_train_data(self, source_data_path, target_data_path=None):
        # 載入 Source Data (用於訓練)
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        print(f"Total Source Dataset Size: {len(self.source_dataset)}")

        # Source 拆分為 Train 和 Val
        source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

        # 載入 Target Data (純粹用於觀察 Baseline 在 Target 上的表現，不參與訓練)
        if target_data_path:
            self.target_dataset = IndoorLocalizationDataset(target_data_path)
            print(f"Total Target Dataset Size (For Validation Only): {len(self.target_dataset)}")
            # 這裡我們不需要拆分 Target Train/Val，因為我們完全不訓練 Target
            # 但為了保持格式一致，我們直接用整個 Target Dataset 做一個 Loader
            self.target_val_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.target_val_loader = None

    def load_test_data(self, test_data_path):
        self.test_dataset = IndoorLocalizationDataset(test_data_path)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
        self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=41)
        self.model = SimpleDNN(self.feature_extractor, self.label_predictor)
        
        # 搬移到 GPU
        self.model.to(self.device)

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_metrics(self):
        self.train_losses, self.train_accuracies = [], []
        self.val_losses, self.val_source_accuracies, self.val_target_accuracies = [], [], []

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            # Training Phase (Only Source)
            train_loss, train_acc = self._run_epoch(self.source_train_loader, training=True)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation Phase
            with torch.no_grad():
                # Validate on Source
                val_loss, val_source_acc = self._run_epoch(self.source_val_loader, training=False)
                
                # Validate on Target (如果有的話，看看沒有 DA 的效果多差)
                val_target_acc = 0.0
                if self.target_val_loader:
                    _, val_target_acc = self._run_epoch(self.target_val_loader, training=False)

                self.val_losses.append(val_loss)
                self.val_source_accuracies.append(val_source_acc)
                self.val_target_accuracies.append(val_target_acc)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Source Acc: {val_source_acc:.4f}, '
                  f'Target Acc (Source Only): {val_target_acc:.4f}')
            
            # Save best model based on Source Validation Loss
            if val_loss < self.best_val_loss:
                print(f'Val Loss improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving model...')
                self.save_model()
                self.best_val_loss = val_loss

    def _run_epoch(self, data_loader, training=False):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for features, labels in data_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            
            _, preds = torch.max(logits, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def plot_training_results(self):
        epochs_list = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs_list, self.train_losses, label='Train Loss', color='blue')
        plt.plot(epochs_list, self.val_losses, label='Val Loss (Source)', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()

        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, self.train_accuracies, label='Train Acc', color='blue')
        plt.plot(epochs_list, self.val_source_accuracies, label='Val Source Acc', color='green')
        if any(self.val_target_accuracies):
            plt.plot(epochs_list, self.val_target_accuracies, label='Target Acc (No DA)', color='red', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig('dnn_training_curve.png')
        plt.close()

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Error: Model file not found at {model_path}")

    def generate_predictions(self, file_path, output_path):
        predictions = {'label': [], 'pred': []}
        self.load_test_data(file_path)
        self.model.eval()
        
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                test_batch = test_batch.to(self.device)
                logits = self.model(test_batch)
                _, preds = torch.max(logits, 1)
                
                predicted_labels = preds + 1
                label = true_label_batch + 1
                
                predictions['label'].extend(label.tolist())
                predictions['pred'].extend(predicted_labels.cpu().tolist())
                
        results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pure DNN Model (Source Only)')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    # Target 這裡只是用來觀察，不會進訓練
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file (For Validation ONLY)')
    parser.add_argument('--test', action='store_true' , help='for test')
    parser.add_argument('--model_path', type=str, default='dnn_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DNN_SourceOnly', help='create new directory to save result')
    parser.add_argument('--unlabeled', action='store_true', help='use unlabeled data from target domain during training')
    parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()
    
    set_seed(args.random_seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 建立資料夾名稱 (Source Only 不需要在意 loss weight 或 unlabeled)
    folder_name = f'SourceOnly_Epoch{args.epoch}'
    work_dir = os.path.join(script_dir, args.work_dir, f'random_seed_{args.random_seed}', folder_name)
    
    dnn_model = HistCorrDNNModel(model_save_path=args.model_path, work_dir=work_dir)

    if args.training_source_domain_data:
        # 訓練模式
        dnn_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data)
        dnn_model.train(num_epochs=args.epoch)
        dnn_model.plot_training_results()
        
    elif args.test:
        # 測試模式
        dnn_model.load_model(args.model_path)
        # 這裡的路徑可能需要根據你的電腦調整，或者也改成參數傳入
        testing_file_paths = [
            r'D:\paper_thesis\Histloc_real\Experiment\data\220318\GalaxyA51\wireless_testing.csv',
            r'D:\paper_thesis\Histloc_real\Experiment\data\231116\GalaxyA51\wireless_testing.csv',
            r'D:\paper_thesis\Histloc_real\Experiment\data\231117\GalaxyA51\wireless_testing.csv'
        ]
        output_paths = ['predictions/220318_results.csv', 'predictions/231116_results.csv', 'predictions/231117_results.csv']
        
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
            
        for testing_file_path, output_path in zip(testing_file_paths, output_paths):
            if os.path.exists(testing_file_path):
                dnn_model.generate_predictions(testing_file_path, output_path)
            else:
                print(f"Warning: Test file not found: {testing_file_path}")
    else:
        print('Please specify --training_source_domain_data')

    os.chdir('..\\..')