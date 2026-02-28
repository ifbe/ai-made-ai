# train_stage0.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import argparse
import glob
import os
from collections import defaultdict
from model import FearGreedWuziqiModel
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS
)

# 创建保存模型的文件夹
os.makedirs("stage0", exist_ok=True)

# 自动选择最佳设备
def get_best_device():
    """自动选择最佳设备：MLX (Apple Silicon) > CUDA > CPU"""
    try:
        if torch.backends.mps.is_available():
            print("✅ 使用 Apple Silicon (MPS)")
            return torch.device("mps")
    except:
        pass
    
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print(f"✅ 使用 NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except:
            print("⚠️ CUDA 可用但兼容性有问题，回退到 CPU")
    
    print("⚠️ 使用 CPU")
    return torch.device("cpu")

def find_latest_checkpoint():
    """查找最新的stage0检查点"""
    checkpoint_files = glob.glob("stage0/wuziqi_stage0_checkpoint_epoch*.pth")
    if not checkpoint_files:
        return None
    
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('epoch')[1].split('.')[0])
            epochs.append((epoch, f))
        except:
            continue
    
    if epochs:
        latest = max(epochs, key=lambda x: x[0])
        return latest[1]
    return None

class Stage0Trainer:
    def __init__(self, model, device, lr=1e-4, start_epoch=1):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.start_epoch = start_epoch
        self.best_acc = 0
        
        if device.type == 'mps':
            print("   MPS 模式: 使用更稳定的设置")
            torch.backends.mps.enable_fallback_to_cpu = True
    
    def save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc
        }
        full_path = os.path.join("stage0", filename)
        torch.save(checkpoint, full_path)
        self.model.to(self.device)
        print(f"   💾 检查点已保存: {full_path}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint.get('best_acc', 0)
            print(f"   ✅ 从 {filename} 恢复训练")
            print(f"      上次训练到 Epoch {checkpoint['epoch']}, 最佳准确率: {self.best_acc:.2%}")
            return True
        return False
    
    def train_step(self, batch):
        boards = []
        players = []
        actions = []
        fear_labels = []
        greed_labels = []
        
        for item in batch:
            # 字典格式读取
            board = item['board']
            player = item['player']
            action = item['move']
            fear_label = item.get('fear_label')
            greed_label = item.get('greed_label')
            
            boards.append(board)
            players.append(player)
            actions.append(action)
            fear_labels.append(fear_label)
            greed_labels.append(greed_label)
        
        boards = torch.tensor(boards, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        players = torch.tensor([0 if p == BLACK else 1 for p in players], 
                       dtype=torch.long, device=self.device)
        
        # 前向传播，获取详细信息
        details = self.model.forward_with_mask(boards, players, return_details=True)
        policy_logits = details['policy']
        
        # 主损失：策略交叉熵
        policy_loss = F.cross_entropy(policy_logits, actions)
        
        total_loss = policy_loss
        
        # 辅助损失1：恐惧分数（如果有标签）
        fear_mask = [f is not None for f in fear_labels]
        if any(fear_mask):
            fear_targets = []
            for i, f in enumerate(fear_labels):
                if f is not None:
                    fear_targets.append(torch.tensor(f, device=self.device))
                else:
                    fear_targets.append(torch.zeros(BOARD_POSITIONS, device=self.device))
            
            fear_targets = torch.stack(fear_targets)
            fear_scores = details['fear_scores']
            fear_loss = F.binary_cross_entropy(fear_scores, fear_targets)
            total_loss += 0.3 * fear_loss
        
        # 辅助损失2：贪婪分数（如果有标签）
        greed_mask = [g is not None for g in greed_labels]
        if any(greed_mask):
            greed_targets = []
            for i, g in enumerate(greed_labels):
                if g is not None:
                    greed_targets.append(torch.tensor(g, device=self.device))
                else:
                    greed_targets.append(torch.zeros(BOARD_POSITIONS, device=self.device))
            
            greed_targets = torch.stack(greed_targets)
            greed_scores = details['greed_scores']
            greed_loss = F.binary_cross_entropy(greed_scores, greed_targets)
            total_loss += 0.4 * greed_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': total_loss.item(), 'policy_loss': policy_loss.item()}
    
    def train_epoch(self, dataset, batch_size=32):
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [dataset[i] for i in batch_indices]
            stats = self.train_step(batch)
            total_loss += stats['loss']
            batch_count += 1
        
        return {'loss': total_loss / batch_count}
    
    def evaluate_accuracy(self, dataset, num_samples=200):
        """只在fear/greed场景上评估"""
        self.model.eval()
        correct = 0
        total = 0
        
        # 筛选fear和greed场景
        unique_scenes = [d for d in dataset if d['scene_type'] in ['fear', 'greed']]
        if len(unique_scenes) < num_samples:
            samples = unique_scenes
        else:
            samples = random.sample(unique_scenes, num_samples)
        
        print(f"    评估场景: fear+greed共 {len(samples)} 个")
        
        with torch.no_grad():
            for item in samples:
                board = item['board']
                player = item['player']
                correct_action = item['move']

                scene_type = item['scene_type']
                fear_label = item.get('fear_label')
                greed_label = item.get('greed_label')
                
                nearby = get_nearby_moves(board, distance=2)
                if not nearby:
                    continue
                
                board_tensor = torch.tensor(board, dtype=torch.long, device=self.device).unsqueeze(0)
                player_tensor = torch.tensor([player], device=self.device)
                
                policy_logits, _ = self.model.forward_with_mask(
                    board_tensor, player_tensor, legal_moves=[nearby]
                )
                
                probs = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                nearby_probs = [(pos, probs[pos]) for pos in nearby]
                
                if nearby_probs:
                    predicted = max(nearby_probs, key=lambda x: x[1])[0]

                    # 其他情况（理论上不会进入这里，因为只评估fear/greed）
                    if predicted == correct_action:
                        correct += 1
                    elif scene_type == 'greed' and greed_label is not None:
                        # 贪婪场景：只要选任意贪婪点就算对
                        greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
                        if predicted in greed_positions:
                            correct += 1
                        else:
                            # 可以打印调试信息
                            pass
                    elif scene_type == 'fear' and fear_label is not None:
                        # 恐惧场景：只要选任意恐惧点就算对
                        fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
                        if predicted in fear_positions:
                            correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0

def load_stage0_dataset(filename="wuziqi_dataset_real.pkl"):
    """加载数据集"""
    print(f"加载数据集: {filename}")
    
    with open(filename, "rb") as f:
        raw_data = pickle.load(f)
    
    print(f"原始数据: {len(raw_data)} 条")
    
    stage0_samples = []
    
    for item in raw_data:
        # 新格式: board, player, action, value, fear_label, greed_label, scene_type
        if len(item) >= 7:
            board, player, action, value, fear_label, greed_label, scene_type = item[:7]
            
            sample = {
                'board': board,
                'player': player,
                'move': action,
                'value': value,
                'scene_type': scene_type
            }
            
            # 添加标签（如果有）
            if fear_label is not None:
                sample['fear_label'] = fear_label
            if greed_label is not None:
                sample['greed_label'] = greed_label
            
            stage0_samples.append(sample)
    
    print(f"Stage0数据集: {len(stage0_samples)} 条")
    
    counts = defaultdict(int)
    fear_count = 0
    greed_count = 0
    player_counts = {BLACK: 0, WHITE: 0}
    
    for s in stage0_samples:
        counts[s['scene_type']] += 1
        player_counts[s['player']] += 1
        if 'fear_label' in s:
            fear_count += 1
        if 'greed_label' in s:
            greed_count += 1
    
    for stype, cnt in counts.items():
        print(f"  {stype}: {cnt} ({cnt/len(stage0_samples):.1%})")
    
    print(f"  玩家分布:")
    print(f"    黑棋回合: {player_counts[BLACK]} ({player_counts[BLACK]/len(stage0_samples):.1%})")
    print(f"    白棋回合: {player_counts[WHITE]} ({player_counts[WHITE]/len(stage0_samples):.1%})")
    
    print(f"  恐惧标签: {fear_count} 个")
    print(f"  贪婪标签: {greed_count} 个")
    
    unique_count = counts.get('fear', 0) + counts.get('greed', 0)
    print(f"  **唯一场景(fear+greed): {unique_count} ({unique_count/len(stage0_samples):.1%})")
    
    return stage0_samples

def main():
    parser = argparse.ArgumentParser(description='五子棋 Stage0 训练')
    parser.add_argument('--continue', '-c', action='store_true', default=True,
                       help='从上次的检查点继续训练')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                       help='指定检查点文件路径')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    args = parser.parse_args()
    
    continue_training = getattr(args, 'continue', True)
    
    print("=" * 70)
    print("五子棋训练 Stage0")
    print("=" * 70)
    
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    print("\n[1/3] 加载数据集...")
    dataset = load_stage0_dataset("wuziqi_dataset_real.pkl")
    
    random.shuffle(dataset)
    split = int(len(dataset) * 0.9)
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    print("\n[2/3] 初始化模型...")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )
    
    trainer = Stage0Trainer(model, device, lr=args.lr)
    
    checkpoint_file = None
    if continue_training:
        if args.checkpoint:
            checkpoint_file = args.checkpoint
        else:
            checkpoint_file = find_latest_checkpoint()
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            trainer.load_checkpoint(checkpoint_file)
        else:
            print("   ⚠️ 未找到检查点，从头开始训练")
    else:
        print("   从头开始训练")
    
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[3/3] 开始训练...")
    print("-" * 90)
    
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        train_stats = trainer.train_epoch(train_data, batch_size=32)
        
        if epoch % 5 == 0:
            val_acc = trainer.evaluate_accuracy(val_data)
            print(f"\nEpoch {epoch:3d}/{args.epochs} | Loss: {train_stats['loss']:.4f} | 唯一场景准确率: {val_acc:.2%}")
            
            if val_acc > trainer.best_acc:
                trainer.best_acc = val_acc
                # 保存最佳模型到stage0文件夹
                torch.save(model.cpu().state_dict(), f"stage0/wuziqi_stage0_best_{val_acc:.0%}_epoch{epoch}.pth")
                model.to(device)
                print(f"          🏆 新最佳模型! 准确率={val_acc:.2%}")
            
            if epoch % 10 == 0:
                trainer.save_checkpoint(epoch, f"wuziqi_stage0_checkpoint_epoch{epoch}.pth")
        
        trainer.scheduler.step()
    
    # 保存最终模型到stage0文件夹
    torch.save(model.cpu().state_dict(), "stage0/wuziqi_stage0_final.pth")
    print(f"\n✅ Stage0训练完成！最佳唯一场景准确率: {trainer.best_acc:.2%}")

if __name__ == "__main__":
    main()
