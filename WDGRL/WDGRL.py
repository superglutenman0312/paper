'''
python .\WDGRL.py --training_source_domain_data D:\Experiment\data\MTLocData\Mall\2021-11-20\wireless_training.csv ^
                --training_target_domain_data D:\Experiment\data\MTLocData\Mall\2022-12-21\wireless_training.csv ^
                --work_dir 211120_221221_WDGRL\\unlabeled\10_10
python WDGRL.py --test --work_dir 211120_221221_WDGRL\\unlabeled\10_10

python WDGRL.py --training_source_domain_data "D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_training.csv" --training_target_domain_data "D:/paper_thesis/Histloc_real/Experiment/data/231117/GalaxyA51/wireless_training.csv" --work_dir "MCSL_unlabeled_251108"
python WDGRL.py --test --work_dir "MCSL_unlabeled_251108"
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

# (DANN_CORR.py 原有)
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # [MODIFIED]
        # x = torch.relu(x) # <-- 這是 DANN_CORR 的做法 (Unbounded)
        x = torch.sigmoid(x) # <-- 這是您成功版本 (modules_wdgrl.py) 的做法 (Bounded [0, 1])
                           # WGAN-GP 需要有界的特徵空間以幫助 Critic 穩定
        return x

# (DANN_CORR.py 原有)
class LabelPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes) # input_size 應為 64

    def forward(self, x):
        x = self.fc1(x)
        return x

# (WDGRL 新增)
# class DomainCritic(nn.Module):
#     def __init__(self, input_size, hidden_size=128): # [MODIFIED] 稍微加寬
#         super(DomainCritic, self).__init__()
#         # [MODIFIED] 增加網路深度
#         self.network = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size // 2), # 新增一層
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, 1)
#         )
#     def forward(self, x):
#         return self.network(x)

class DomainCritic(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(DomainCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# (DANN_CORR.py 原有, 略微修改)
class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, label_predictor):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        features = self.feature_extractor(x)
        labels = self.label_predictor(features)
        return features, labels

# (DANN_CORR.py 原有)
class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# (DANN_CORR.py 風格, WDGRL 核心)
class WDGRLModel:
    # [MODIFIED] 更改預設 lr
    def __init__(self, model_save_path='saved_model.pth', lambda_val=10.0, gamma_val=10.0, critic_steps=5, lr=0.0001, work_dir=None):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.batch_size = 32
        self.lambda_val = lambda_val 
        self.gamma_val = gamma_val   
        self.critic_steps = critic_steps 
        self.lr = lr # [MODIFIED] 將使用傳入的 0.0001
        
        self.input_size = 7
        self.feature_extractor_neurons = [32, 16]
        self.num_classes = 41

        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')

    # (DANN_CORR.py 原有, MODIFIED: 加入 drop_last=True 修正 Runtime Error)
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
        # [MODIFIED] 加入 drop_last=True
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False, drop_last=True)

        target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
        # [MODIFIED] 加入 drop_last=True
        self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False, drop_last=True)

    # (DANN_CORR.py 原有)
    def load_test_data(self, test_data_path):
        self.test_dataset = IndoorLocalizationDataset(test_data_path)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

    # (修改)
    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1]).to(self.device)
        self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=self.num_classes).to(self.device)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor).to(self.device)
        
        self.domain_critic = DomainCritic(input_size=self.feature_extractor_neurons[1], hidden_size=100).to(self.device)

    # (修改)
    def _initialize_optimizer(self):
        # [MODIFIED] 
        # 1. 使用 self.lr (將設為 0.0001)
        # 2. 加入 WGAN-GP 推薦的 betas=(0.5, 0.9)
        self.optimizer_main = optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.label_predictor.parameters()), 
            lr=self.lr,
            betas=(0.5, 0.9) # <-- 參照成功版本的關鍵參數
        )
        self.optimizer_critic = optim.Adam(
            self.domain_critic.parameters(), 
            lr=self.lr,
            betas=(0.5, 0.9) # <-- 參照成功版本的關鍵參數
        )
        
        self.domain_criterion = nn.CrossEntropyLoss()

    # (修改)
    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.wasserstein_dists, self.grad_penalties = [], [], [], []
        # [MODIFIED] 移除 total_accuracies，因為我們現在要看 S 和 T
        self.source_accuracies, self.target_accuracies = [], [] 
        self.val_total_losses, self.val_label_losses, self.val_wasserstein_dists, self.val_grad_penalties = [], [], [], []
        # [MODIFIED] 移除 val_total_accuracies
        self.val_source_accuracies, self.val_target_accuracies = [], []

    # (WDGRL 新增)
    def _gradient_penalty(self, h_s, h_t):
        batch_size = h_s.size(0)
        epsilon = torch.rand(batch_size, 1).to(self.device)
        epsilon = epsilon.expand_as(h_s)
        
        h_hat = (epsilon * h_s + (1 - epsilon) * h_t).requires_grad_(True)
        
        critic_output = self.domain_critic(h_hat)
        
        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=h_hat,
            grad_outputs=torch.ones_like(critic_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        grad_penalty = ((grad_norm - 1) ** 2).mean() 
        
        return grad_penalty

    # (DANN_CORR.py 原有)
    def train(self, num_epochs=100, unlabeled=False):
        for epoch in range(num_epochs):
            self.domain_adaptation_model.train()
            self.domain_critic.train()
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.wasserstein_dists.append(loss_list[2])
            self.grad_penalties.append(loss_list[3])
            # [MODIFIED] 儲存 S 和 T 準確率，移除平均值
            # self.total_accuracies.append(acc_list[0])
            self.source_accuracies.append(acc_list[1])
            self.target_accuracies.append(acc_list[2])

            self.domain_adaptation_model.eval()
            self.domain_critic.eval()
            with torch.no_grad():
                val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

                self.val_total_losses.append(val_loss_list[0])
                self.val_label_losses.append(val_loss_list[1])
                self.val_wasserstein_dists.append(val_loss_list[2])
                # [MODIFIED] 儲存 S 和 T 準確率，移除平均值
                # self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
            
            val_gp_str = f", val_wd_dist: {self.val_wasserstein_dists[-1]:.4f}" if not unlabeled else ""
            
            # [MODIFIED] *** 更改日誌輸出 ***
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, wd_dist: {self.wasserstein_dists[-1]:.4f}, grad_pen: {self.grad_penalties[-1]:.4f}, acc_S: {self.source_accuracies[-1]:.4f}, acc_T: {self.target_accuracies[-1]:.4f},\n'
                  f'val_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}{val_gp_str}, val_acc_S: {self.val_source_accuracies[-1]:.4f}, val_acc_T: {self.val_target_accuracies[-1]:.4f}')
            
            # [MODIFIED] 根據 val_label_loss 保存模型 (這是您之前版本成功的策略)
            if self.val_label_losses[-1] < self.best_val_total_loss:
                print(f'val_label_loss: {self.val_label_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = self.val_label_losses[-1]

    # (重寫)
    def _run_epoch(self, data_loader, training=False, unlabeled=False):
        source_correct_predictions, source_total_samples = 0, 0
        target_correct_predictions, target_total_samples = 0, 0
        
        source_iter = cycle(data_loader[0])
        target_iter = cycle(data_loader[1])
        # [MODIFIED] 確保使用較小的 loader 長度，避免 drop_last=True 後仍出錯
        num_batches = min(len(data_loader[0]), len(data_loader[1])) 

        loss_list = [0, 0, 0, 0] 
        acc_list = [0, 0, 0] # [MODIFIED] 雖然回傳值是 3 個, 但我們只會用後 2 個

        if num_batches == 0:
             print("Warning: DataLoader length is zero, skipping epoch.")
             return loss_list, acc_list

        for i in range(num_batches):
            source_features, source_labels = next(source_iter)
            target_features, target_labels = next(target_iter)

            source_features, source_labels = source_features.to(self.device), source_labels.to(self.device)
            target_features, target_labels = target_features.to(self.device), target_labels.to(self.device)
            
            # [MODIFIED] 確保 batch size 一致 (drop_last=True 應該已處理，此為雙重保險)
            if source_features.size(0) != target_features.size(0):
                continue

            if training:
                # --- 步驟 1: 訓練 Domain Critic (n 步) ---
                for _ in range(self.critic_steps):
                    self.optimizer_critic.zero_grad()
                    
                    with torch.no_grad(): # [MODIFIED] 確保 FE 在 Critic 訓練時不計算梯度
                        h_s_crit = self.feature_extractor(source_features).detach()
                        h_t_crit = self.feature_extractor(target_features).detach()
                    
                    grad_penalty = self._gradient_penalty(h_s_crit, h_t_crit)
                    
                    crit_s_out = self.domain_critic(h_s_crit)
                    crit_t_out = self.domain_critic(h_t_crit)
                    wasserstein_dist_crit = crit_s_out.mean() - crit_t_out.mean() 
                    
                    loss_critic = -wasserstein_dist_crit + self.gamma_val * grad_penalty
                    
                    loss_critic.backward()
                    self.optimizer_critic.step()
                
                # --- 步驟 2: 訓練 Feature Extractor 和 Label Predictor ---
                self.optimizer_main.zero_grad()
                
                h_s, labels_s_pred = self.domain_adaptation_model(source_features)
                h_t, labels_t_pred = self.domain_adaptation_model(target_features)
                
                label_loss_source = self.domain_criterion(labels_s_pred, source_labels)
                label_loss_target = self.domain_criterion(labels_t_pred, target_labels)
                
                if unlabeled:
                    label_loss = label_loss_source
                else:
                    label_loss = (label_loss_source + label_loss_target) / 2
                
                crit_s_main = self.domain_critic(h_s)
                crit_t_main = self.domain_critic(h_t)
                # [MODIFIED] 修正 WDGRL 論文 (Algorithm 1) 的對抗損失
                # Feature Extractor (G_g) 的目標是最小化 (L_c + lambda * L_wd)
                # L_wd = E[f_w(h_s)] - E[f_w(h_t)]
                wasserstein_dist_main = crit_s_main.mean() - crit_t_main.mean() 
                
                total_loss = label_loss + self.lambda_val * wasserstein_dist_main
                
                total_loss.backward()
                self.optimizer_main.step()

                loss_list = [total_loss.item(), label_loss.item(), wasserstein_dist_main.item(), grad_penalty.item()]

            else: 
                h_s, labels_s_pred = self.domain_adaptation_model(source_features)
                h_t, labels_t_pred = self.domain_adaptation_model(target_features)

                label_loss_source = self.domain_criterion(labels_s_pred, source_labels)
                label_loss_target = self.domain_criterion(labels_t_pred, target_labels)
                
                if unlabeled:
                    label_loss = label_loss_source
                    total_loss = label_loss 
                    wasserstein_dist_main = torch.tensor(0.0)
                else:
                    label_loss = (label_loss_source + label_loss_target) / 2
                    crit_s_val = self.domain_critic(h_s)
                    crit_t_val = self.domain_critic(h_t)
                    wasserstein_dist_main = crit_s_val.mean() - crit_t_val.mean()
                    total_loss = label_loss + self.lambda_val * wasserstein_dist_main

                loss_list = [total_loss.item(), label_loss.item(), wasserstein_dist_main.item(), 0.0]

            _, source_preds = torch.max(labels_s_pred, 1)
            source_correct_predictions += (source_preds == source_labels).sum().item()
            source_total_samples += source_labels.size(0)
            
            # [MODIFIED] 即使在 unlabeled=True, 仍然計算 target 準確率 (僅用於 "評估", 不用於 "訓練")
            _, target_preds = torch.max(labels_t_pred, 1)
            target_correct_predictions += (target_preds == target_labels).sum().item()
            target_total_samples += target_labels.size(0)

        # [MODIFIED] 避免
        if source_total_samples == 0: source_total_samples = 1 
        if target_total_samples == 0: target_total_samples = 1

        source_accuracy = source_correct_predictions / source_total_samples
        target_accuracy = target_correct_predictions / target_total_samples
        
        # [MODIFIED] 雖然回傳 3 個值, 但第一個平均值我們不再使用了
        acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
        
        return loss_list, acc_list 

    # (DANN_CORR.py 原有)
    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    # (修改)
    def plot_training_results(self):
        epochs_list = np.arange(0, len(self.total_losses), 1)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs_list, self.label_losses, label='Label Loss', color='blue')
        plt.plot(epochs_list, self.val_label_losses, label='Val Label Loss', color='darkorange')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Label Predictor Training Loss')

        # [MODIFIED] *** 更改圖表繪製 ***
        plt.subplot(2, 2, 2)
        plt.plot(epochs_list, self.source_accuracies, label='Train Acc (Source)', color='blue', linestyle='-')
        plt.plot(epochs_list, self.target_accuracies, label='Train Acc (Target)', color='cyan', linestyle='--')
        plt.plot(epochs_list, self.val_source_accuracies, label='Val Acc (Source)', color='red', linestyle='-')
        plt.plot(epochs_list, self.val_target_accuracies, label='Val Acc (Target)', color='orange', linestyle='--')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(fontsize='small'); plt.title('Training Accuracy (Source vs Target)')


        plt.subplot(2, 2, 3)
        plt.plot(epochs_list, self.wasserstein_dists, label='Wasserstein Distance', color='blue')
        plt.plot(epochs_list, self.val_wasserstein_dists, label='Val Wasserstein Distance', color='darkorange')
        plt.xlabel('Epochs'); plt.ylabel('Distance'); plt.legend(); plt.title('Wasserstein Distance (Critic View)')

        plt.subplot(2, 2, 4)
        plt.plot(epochs_list, self.grad_penalties, label='Gradient Penalty', color='green')
        plt.xlabel('Epochs'); plt.ylabel('Penalty'); plt.legend(); plt.title('Gradient Penalty (Training)')

        plt.suptitle('WDGRL Training Curve'); plt.tight_layout(); plt.savefig('loss_and_accuracy_wdgrl.png')

    # (DANN_CORR.py 原有)
    def save_model_architecture(self, file_path='model_architecture'):
        dummy_input = torch.randn(1, 1033)
        # graph = make_dot(self.domain_adaptation_model(dummy_input.to(self.device)), params=dict(self.domain_adaptation_model.named_parameters()))
        # graph.render(file_path, format='png')
        print(f"Model architecture saved as {file_path}")

    # (DANN_CORR.py 原有)
    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.domain_adaptation_model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Error: Model file not found at {model_path}")

    # (DANM_CORR.py 原有)
    def predict(self, features):
        self.domain_adaptation_model.eval()
        with torch.no_grad():
            features, labels_pred = self.domain_adaptation_model(features.to(self.device))
        return labels_pred.cpu()

    # (DANN_CORR.py 原有)
    def generate_predictions(self, file_path, output_path):
        predictions = {'label': [], 'pred': []}
        self.load_test_data(file_path)
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                labels_pred = self.predict(test_batch)
                _, preds = torch.max(labels_pred, 1)
                predicted_labels = preds + 1
                label = true_label_batch + 1
                predictions['label'].extend(label.tolist())
                predictions['pred'].extend(predicted_labels.tolist())
        results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
        results.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train WDGRL Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--test', action='store_true' , help='for test')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='WDGRL_CORR', help='create new directory to save result')
    parser.add_argument('--lambda_val', type=float, default=10.0, help='Weight for Wasserstein distance loss (lambda)') 
    parser.add_argument('--gamma_val', type=float, default=10.0, help='Weight for Gradient Penalty (gamma)') 
    parser.add_argument('--critic_steps', type=int, default=5, help='Number of critic steps per extractor step (n)') 
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    # [MODIFIED] 新增學習率參數
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizers')

    args = parser.parse_args()
    
    lambda_val = args.lambda_val
    gamma_val = args.gamma_val
    critic_steps = args.critic_steps
    epoch = args.epochs
    lr = args.lr # [MODIFIED] 使用 args 的 lr
    
    unlabeled = False
    
    data_drop_out_list = np.arange(0.9) 
    # data_drop_out_list = np.arange(0.0, 0.05, 0.1) 
    
    for data_drop_out in data_drop_out_list:
        wdgrl_model = WDGRLModel(
            model_save_path=args.model_path, 
            lambda_val=lambda_val, 
            gamma_val=gamma_val, 
            critic_steps=critic_steps,
            lr=lr, # [MODIFIED] 傳入正確的學習率 0.0001
            work_dir=f'{args.work_dir}_{data_drop_out:.1f}'
        )
        # wdgrl_model.save_model_architecture()
        
        if args.training_source_domain_data and args.training_target_domain_data:
            wdgrl_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            wdgrl_model.train(num_epochs=epoch, unlabeled=unlabeled)
            wdgrl_model.plot_training_results()
        elif args.test:
            wdgrl_model.load_model(args.model_path)
            # testing_file_paths = [
            #             r'D:\paper_thesis\Histloc_try\mall_data\Mall\2021-11-20\wireless_testing.csv',
            #             r'D:\paper_thesis\Histloc_try\mall_data\Mall\2022-12-21\wireless_testing.csv'
            #         ]

            # MCSL 測試資料路徑
            testing_file_paths = [
                        r'D:/paper_thesis/Histloc_real/Experiment/data/231116/GalaxyA51/wireless_testing.csv',
                        r'D:/paper_thesis/Histloc_real/Experiment/data/231117/GalaxyA51/wireless_testing.csv'
                    ]
            output_paths = ['predictions/211120_results.csv', 'predictions/221221_results.csv']
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            for testing_file_path, output_path in zip(testing_file_paths, output_paths):
                wdgrl_model.generate_predictions(testing_file_path, output_path)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')