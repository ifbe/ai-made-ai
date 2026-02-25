import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本所在目录

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
from collections import defaultdict
from model import FearGreedWuziqiModel
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS
)

def get_best_device():
    """自动选择最佳设备：MLX (Apple Silicon) > CUDA > CPU"""
    try:
        if torch.backends.mps.is_available():
            print("✅ 使用 Apple Silicon (MPS)")
            return torch.device("mps")
    except:
        pass
    
    if torch.cuda.is_available():
        print("✅ 使用 NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    
    print("⚠️ 使用 CPU")
    return torch.device("cpu")

class Stage1Trainer:
    def __init__(self, model, device, lr=5e-5):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        if device.type == 'mps':
            print("   MPS 模式: 使用更稳定的设置")
            torch.backends.mps.enable_fallback_to_cpu = True
    
    def train_step(self, batch):
        boards = []
        turns = []
        actions = []
        rewards = []
        legal_moves_list = []
        
        for item in batch:
            boards.append(item['board'])
            turns.append(item['turn'])
            actions.append(item['move'])
            rewards.append(item['reward'])
            
            nearby = get_nearby_moves(item['board'], distance=2)
            if not nearby:
                center = BOARD_SIZE // 2
                nearby = [center * BOARD_SIZE + center]
            legal_moves_list.append(nearby)
        
        boards = torch.tensor(boards, dtype=torch.long, device=self.device)
        turns = torch.tensor(turns, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        
        policy_logits, values = self.model.forward_with_mask(
            boards, turns, legal_moves=legal_moves_list
        )
        
        policy_logits = torch.clamp(policy_logits, -20, 20)
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        advantages = rewards - values.squeeze().detach()
        advantages = torch.clamp(advantages, -1, 1)
        
        policy_loss = -(action_log_probs * advantages).mean()
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        entropy = 0
        for b in range(len(batch)):
            valid_indices = legal_moves_list[b]
            if valid_indices:
                valid_logits = policy_logits[b][valid_indices]
                valid_probs = F.softmax(valid_logits, dim=-1)
                valid_log_probs = F.log_softmax(valid_logits, dim=-1)
                entropy += -(valid_probs * valid_log_probs).sum()
        entropy = entropy / len(batch)
        entropy_bonus = 0.0005 * entropy
        
        total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
        
        if torch.isnan(total_loss):
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'avg_reward': rewards.mean().item()
            }
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'avg_reward': rewards.mean().item()
        }
    
    def train_epoch(self, dataset, batch_size=32):
        self.model.train()
        epoch_stats = defaultdict(float)
        batch_count = 0
        valid_batches = 0
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [dataset[i] for i in batch_indices]
            stats = self.train_step(batch)
            
            if stats['total_loss'] != 0.0:
                for k, v in stats.items():
                    epoch_stats[k] += v
                valid_batches += 1
            batch_count += 1
        
        if valid_batches > 0:
            for k in epoch_stats:
                epoch_stats[k] /= valid_batches
        
        return epoch_stats
    
    def evaluate(self, num_games=5):
        """评估模型对战能力"""
        self.model.eval()
        
        wins = {BLACK: 0, WHITE: 0}
        draws = 0
        
        for _ in range(num_games):
            board = [EMPTY] * BOARD_POSITIONS
            current = BLACK
            
            center = BOARD_SIZE // 2
            first_move = center * BOARD_SIZE + center
            board[first_move] = current
            current = WHITE
            
            move_count = 1
            
            while move_count < 60:
                nearby = get_nearby_moves(board, distance=2)
                if not nearby:
                    break
                
                with torch.no_grad():
                    board_tensor = torch.tensor(board, dtype=torch.long, device=self.device).unsqueeze(0)
                    turn_tensor = torch.tensor([0 if current == BLACK else 1], device=self.device)
                    
                    policy_logits, _ = self.model.forward_with_mask(
                        board_tensor, turn_tensor, legal_moves=[nearby]
                    )
                    
                    probs = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                    nearby_probs = [(pos, probs[pos]) for pos in nearby]
                    move = max(nearby_probs, key=lambda x: x[1])[0]
                
                board[move] = current
                move_count += 1
                
                if check_win(board, current):
                    wins[current] += 1
                    break
                
                current = 3 - current
            
            if move_count >= 60:
                draws += 1
        
        return {
            'black_wins': wins[BLACK],
            'white_wins': wins[WHITE],
            'draws': draws,
            'win_rate': (wins[BLACK] + wins[WHITE]) / num_games
        }

def load_all_dataset(filename="wuziqi_dataset_real.pkl"):
    """加载全部数据集"""
    print(f"加载数据集: {filename}")
    
    with open(filename, "rb") as f:
        raw_data = pickle.load(f)
    
    print(f"原始数据: {len(raw_data)} 条")
    
    all_samples = []
    
    for item in raw_data:
        try:
            if len(item) >= 6:
                board, action, value, fear_label, greed_label, scene_type = item[:6]
            else:
                board, action, value = item[:3]
                scene_type = 'normal'
            
            if scene_type == 'fear':
                reward = -0.2
            elif scene_type == 'greed':
                reward = 0.2
            elif scene_type == 'mixed':
                reward = 0.1
            else:
                reward = np.clip(value * 0.1, -0.1, 0.1) if value is not None else 0.0
            
            black_count = sum(1 for x in board if x == BLACK)
            white_count = sum(1 for x in board if x == WHITE)
            
            if black_count > white_count:
                turn = 1
            else:
                turn = 0
            
            all_samples.append({
                'board': board,
                'turn': turn,
                'move': action,
                'reward': reward,
                'scene_type': scene_type
            })
        except:
            continue
    
    print(f"总样本数: {len(all_samples)}")
    
    counts = defaultdict(int)
    rewards = []
    for s in all_samples:
        counts[s['scene_type']] += 1
        rewards.append(s['reward'])
    
    for stype, cnt in counts.items():
        print(f"  {stype}: {cnt} ({cnt/len(all_samples):.1%})")
    
    print(f"  奖励范围: {min(rewards):.3f} ~ {max(rewards):.3f}")
    
    return all_samples

def main():
    print("=" * 70)
    print("五子棋训练 Stage1 - 加载Stage0模型继续训练")
    print("=" * 70)
    
    # 自动选择设备
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    # 加载数据集
    print("\n[1/4] 加载数据集...")
    dataset = load_all_dataset("wuziqi_dataset_real.pkl")
    
    random.shuffle(dataset)
    split = int(len(dataset) * 0.9)
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 创建模型
    print("\n[2/4] 初始化模型...")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    ).to(device)
    
    # 尝试加载Stage0模型
    try:
        # 加载时先放到CPU，再移到目标设备
        state_dict = torch.load("wuziqi_stage0_final.pth", map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("✅ 成功加载 Stage0 模型")
    except Exception as e:
        print(f"⚠️ 加载失败: {e}, 使用随机初始化")
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = Stage1Trainer(model, device, lr=2e-5)
    
    # 训练
    print("\n[3/4] 开始微调...")
    print("-" * 90)
    
    best_win_rate = 0
    
    for epoch in range(1, 21):
        train_stats = trainer.train_epoch(train_data, batch_size=32)
        
        if epoch % 2 == 0:
            eval_results = trainer.evaluate(num_games=5)
            
            print(f"\nEpoch {epoch:3d} | Loss: {train_stats.get('total_loss', 0):.4f}")
            print(f"  对战: 黑胜={eval_results['black_wins']}, "
                  f"白胜={eval_results['white_wins']}, "
                  f"平局={eval_results['draws']}, "
                  f"胜率={eval_results['win_rate']:.2%}")
            
            if eval_results['win_rate'] > best_win_rate:
                best_win_rate = eval_results['win_rate']
                # 保存时转为CPU张量
                torch.save(model.cpu().state_dict(), f"wuziqi_stage1_best_{best_win_rate:.0%}_epoch{epoch}.pth")
                model.to(device)  # 移回原设备
                print(f"          🏆 新最佳模型! 胜率={best_win_rate:.2%}")
        
        trainer.scheduler.step()
    
    # 保存最终模型
    torch.save(model.cpu().state_dict(), "wuziqi_stage1_final.pth")
    
    print("\n[4/4] 训练完成！")
    print("=" * 70)
    print(f"Stage1最佳胜率: {best_win_rate:.2%}")
    print("=" * 70)

if __name__ == "__main__":
    main()