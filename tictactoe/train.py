# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random
import numpy as np
from collections import defaultdict
from model import FearGreedModel
from game import get_legal_moves

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        self.stats = defaultdict(list)
    
    def train_epoch(self, dataset, batch_size=64):
        self.model.train()
        epoch_losses = defaultdict(float)
        batch_count = 0
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [dataset[i] for i in batch_indices]
            
            boards = []
            actions = []
            values = []
            
            for item in batch:
                board, action, value = item[:3]
                boards.append(board)
                actions.append(action)
                values.append(value if value is not None else 0.0)
            
            boards = torch.tensor(boards, dtype=torch.long, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            values = torch.tensor(values, dtype=torch.float, device=self.device)
            
            policy, pred_values = self.model(boards)
            
            policy_loss = self.policy_criterion(policy, actions)
            value_loss = self.value_criterion(pred_values.squeeze(), values)
            
            total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['policy'] += policy_loss.item()
            epoch_losses['value'] += value_loss.item()
            batch_count += 1
        
        for k in epoch_losses:
            epoch_losses[k] /= batch_count
        
        return epoch_losses
    
    def evaluate(self, dataset, num_samples=200):
        self.model.eval()
        
        correct = 0
        total = 0
        
        samples = random.sample(dataset, min(num_samples, len(dataset)))
        
        with torch.no_grad():
            for item in samples:
                board, correct_action, _ = item[:3]
                
                board_tensor = torch.tensor(board, dtype=torch.long, device=self.device).unsqueeze(0)
                policy, _ = self.model(board_tensor)
                
                legals = get_legal_moves(board)
                probs = F.softmax(policy[0], dim=-1).cpu().numpy()
                
                for i in range(9):
                    if i not in legals:
                        probs[i] = 0
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    if probs.argmax() == correct_action:
                        correct += 1
                total += 1
        
        return correct / total if total > 0 else 0

def main():
    print("=" * 70)
    print("恐惧与贪婪模型训练 - 优先级融合版")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    try:
        with open("fear_greed_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
        print(f"\n加载数据集: {len(dataset)} 条")
        
        random.shuffle(dataset)
        split = int(len(dataset) * 0.8)
        train_data = dataset[:split]
        test_data = dataset[split:]
        
        print(f"  训练集: {len(train_data)} 条")
        print(f"  测试集: {len(test_data)} 条")
        
    except FileNotFoundError:
        print("错误: 找不到 fear_greed_dataset.pkl")
        print("请先运行 generate.py 生成数据集")
        return
    except Exception as e:
        print(f"错误: {e}")
        return
    
    model = FearGreedModel().to(device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, device)
    
    print("\n开始训练 (200轮)...")
    print("-" * 90)
    
    best_acc = 0
    
    for epoch in range(1, 201):
        losses = trainer.train_epoch(train_data, batch_size=64)
        
        if epoch % 10 == 0 or epoch == 1:
            acc = trainer.evaluate(test_data)
            
            print(f"\nEpoch {epoch:3d} | "
                  f"LR: {trainer.scheduler.get_last_lr()[0]:.6f} | "
                  f"Loss: 总={losses['total']:.4f} "
                  f"策略={losses['policy']:.4f} "
                  f"价值={losses['value']:.4f} | "
                  f"准确率: {acc:.2%}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"model_best_{acc:.0%}_epoch{epoch}.pth")
                print(f"          🏆 新最佳模型!")
        
        trainer.scheduler.step()
    
    print("\n" + "=" * 70)
    print(f"训练完成！最佳准确率: {best_acc:.2%}")
    print("=" * 70)

if __name__ == "__main__":
    main()
