# train_stage2_selfplay.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import glob
import os
import json
import time
import argparse
from collections import defaultdict
from datetime import datetime
from model import FearGreedWuziqiModel
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS, pos_to_str
)

# 创建保存模型的文件夹
os.makedirs("stage2", exist_ok=True)

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

def find_best_stage1_model():
    """查找stage1的最佳模型（取epoch最大的）"""
    # 优先找stage1的checkpoint
    checkpoint_files = glob.glob("stage1/wuziqi_stage1_minimax_checkpoint_epoch*.pth")
    if checkpoint_files:
        best_model = None
        max_epoch = 0
        for f in checkpoint_files:
            try:
                epoch_str = f.split('epoch')[1].split('.')[0]
                epoch = int(epoch_str)
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_model = f
            except:
                continue
        if best_model:
            print(f"   找到Stage1模型: epoch {max_epoch}")
            return best_model
    
    # 如果没有，找stage1的best文件
    best_files = glob.glob("stage1/wuziqi_stage1_minimax_best_*.pth")
    if best_files:
        return max(best_files, key=os.path.getctime)
    
    return None

def find_latest_checkpoint():
    """查找最新的stage2检查点"""
    checkpoint_files = glob.glob("stage2/wuziqi_stage2_checkpoint_epoch*.pth")
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

def print_game_board(moves_history, title=None):
    """
    打印一局棋的完整棋盘，用数字显示落子顺序
    黑子：白色数字在黑色背景
    白子：黑色数字在白色背景
    """
    if title:
        print(f"\n{title}")
    
    board_state = [EMPTY] * BOARD_POSITIONS
    move_numbers = [0] * BOARD_POSITIONS
    
    for step, (board, player, move) in enumerate(moves_history):
        board_state[move] = player
        move_numbers[move] = step + 1
    
    print("\n   ", end="")
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f"│ {i} ", end="")
        else:
            print(f"│ {chr(ord('a') + (i-10))} ", end="")
    print("│")
    
    print("   " + "├───" * BOARD_SIZE + "┤")
    
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
        
        for j in range(BOARD_SIZE):
            idx = i * BOARD_SIZE + j
            player = board_state[idx]
            move_num = move_numbers[idx]
            
            if player == BLACK:
                print(f"│\033[97;40m{move_num:^3}\033[0m", end="")
            elif player == WHITE:
                print(f"│\033[30;107m{move_num:^3}\033[0m", end="")
            else:
                print(f"│   ", end="")
        
        print("│")
        
        if i < BOARD_SIZE - 1:
            print("   " + "├───" * BOARD_SIZE + "┤")
    
    print("   " + "└───" * BOARD_SIZE + "┘")
    
    print("\n图例: \033[97;40m 数字 \033[0m = 黑子, \033[30;107m 数字 \033[0m = 白子")

def save_games_to_file(games_list, filename="history_stage2_selfplay.json"):
    """批量保存棋局到文件"""
    all_games = []
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_games = json.load(f)
        except:
            all_games = []
    
    all_games.extend(games_list)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_games, f, ensure_ascii=False, indent=2)

class Stage2SelfPlayTrainer:
    def __init__(self, model, device, lr=1e-5, start_epoch=1):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.start_epoch = start_epoch
        self.best_win_rate = 0
        
        if device.type == 'mps':
            print("   MPS 模式: 使用更稳定的设置")
            torch.backends.mps.enable_fallback_to_cpu = True
    
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_win_rate': self.best_win_rate
        }
        full_path = os.path.join("stage2", filename)
        torch.save(checkpoint, full_path)
        self.model.to(self.device)
        print(f"   💾 检查点已保存: {full_path}")
    
    def load_checkpoint(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_win_rate = checkpoint.get('best_win_rate', 0)
            print(f"   ✅ 从 {filename} 恢复训练")
            print(f"      上次训练到 Epoch {checkpoint['epoch']}, 最佳胜率: {self.best_win_rate:.2%}")
            return True
        return False
    
    def generate_self_play_games(self, num_games=20, max_moves=60, save_games=True):
        """
        自我对弈生成训练数据
        每局随机决定先手方，让模型同时学习先手和后手策略
        """
        games_data = []
        games_records = []
        black_win_count = 0
        white_win_count = 0
        draw_count = 0
        first_game_board = None
        first_game_moves = 0
        first_game_winner = None
        
        for game_idx in range(num_games):
            # 随机决定本局谁先手
            black_first = random.choice([True, False])  # True=黑先, False=白先
            
            board = [EMPTY] * BOARD_POSITIONS
            game_steps = []
            moves_history = []
            
            # 设置当前玩家
            if black_first:
                current = BLACK
                first_player = "black"
            else:
                current = WHITE
                first_player = "white"
            
            # 第一步下中心（先手方下）
            center = BOARD_SIZE // 2
            first_move = center * BOARD_SIZE + center
            board[first_move] = current
            game_steps.append((board.copy(), current, first_move))
            moves_history.append((board.copy(), current, first_move))
            
            # 切换到另一方
            current = 3 - current
            move_count = 1
            
            while move_count < max_moves:
                with torch.no_grad():
                    board_tensor = torch.tensor(board, dtype=torch.long, device=self.device).unsqueeze(0)
                    player_tensor = torch.tensor([0 if current == BLACK else 1], device=self.device)
                    policy_logits, _ = self.model.forward_with_mask(board_tensor, player_tensor)
                    
                    # 中等温度，平衡探索与利用
                    temperature = 0.5
                    probs = F.softmax(policy_logits[0] / temperature, dim=-1).cpu().numpy()
                    
                    legals = get_legal_moves(board)
                    if not legals:
                        break
                    
                    # 只考虑合法位置
                    valid_probs = probs[legals]
                    if valid_probs.sum() > 0:
                        valid_probs = valid_probs / valid_probs.sum()
                        move = np.random.choice(legals, p=valid_probs)
                    else:
                        move = random.choice(legals)
                
                board[move] = current
                game_steps.append((board.copy(), current, move))
                moves_history.append((board.copy(), current, move))
                move_count += 1
                
                if check_win(board, current):
                    break
                
                current = 3 - current
            
            # 确定胜负
            if check_win(board, BLACK):
                winner = 1
                winner_str = "black"
                if black_first:
                    black_win_count += 1
                else:
                    white_win_count += 1
            elif check_win(board, WHITE):
                winner = -1
                winner_str = "white"
                if not black_first:
                    black_win_count += 1
                else:
                    white_win_count += 1
            else:
                winner = 0
                winner_str = "draw"
                draw_count += 1
            
            # 保存第一局用于打印
            if game_idx == 0:
                first_game_board = moves_history
                first_game_moves = move_count
                first_game_winner = winner_str
            
            # 为每一步计算奖励（只有胜负局才学习）
            if winner != 0:
                gamma = 0.95
                T = len(game_steps)
                
                for t, (_, player, move) in enumerate(game_steps):
                    steps_to_end = T - t
                    reward = winner if player == BLACK else -winner
                    reward = reward * (gamma ** steps_to_end)
                    
                    games_data.append({
                        'board': game_steps[t][0],
                        'player': 0 if player == BLACK else 1,
                        'move': move,
                        'reward': reward,
                        'winner': winner
                    })
            
            # 保存棋局记录
            if save_games:
                games_records.append({
                    'timestamp': datetime.now().isoformat(),
                    'epoch': self.start_epoch,
                    'game_id': game_idx,
                    'first_player': first_player,
                    'winner': winner_str,
                    'moves': [(pos_to_str(move), 'black' if player == BLACK else 'white') 
                             for _, player, move in moves_history]
                })
        
        if save_games and games_records:
            save_games_to_file(games_records)
        
        total_games = black_win_count + white_win_count + draw_count
        win_rate = (black_win_count + white_win_count) / total_games if total_games > 0 else 0
        
        print(f"   对局结果: 黑胜={black_win_count}, 白胜={white_win_count}, 平局={draw_count}, 胜率={win_rate:.2%}")
        print(f"   有效训练样本: {len(games_data)} 个")
        
        return games_data, first_game_board, first_game_moves, first_game_winner
    
    def train_step(self, batch):
        boards = []
        players = []
        actions = []
        rewards = []
        legal_moves_list = []
        
        for item in batch:
            board = item['board']
            player = item['player']
            action = item['move']
            reward = item['reward']
            
            boards.append(board)
            players.append(player)
            actions.append(action)
            rewards.append(reward)
            
            nearby = get_nearby_moves(board, distance=2)
            if not nearby:
                center = BOARD_SIZE // 2
                nearby = [center * BOARD_SIZE + center]
            legal_moves_list.append(nearby)
        
        boards = torch.tensor(boards, dtype=torch.long, device=self.device)
        players = torch.tensor(players, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        
        policy_logits, values = self.model.forward_with_mask(
            boards, players, legal_moves=legal_moves_list
        )
        
        policy_logits = torch.clamp(policy_logits, -20, 20)
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        advantages = rewards - values.squeeze()
        advantages = torch.clamp(advantages, -0.2, 0.2)
        
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(-1), rewards)
        
        entropy = 0
        for b in range(len(batch)):
            valid_indices = legal_moves_list[b]
            if valid_indices:
                valid_logits = policy_logits[b][valid_indices]
                valid_probs = F.softmax(valid_logits, dim=-1)
                valid_log_probs = F.log_softmax(valid_logits, dim=-1)
                entropy += -(valid_probs * valid_log_probs).sum()
        entropy = entropy / len(batch)
        entropy_bonus = 0.0002 * entropy
        
        total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
        total_loss = torch.clamp(total_loss, min=0.01)
        
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
    
    def train_epoch(self, games_per_epoch=20, batch_size=32):
        """每个epoch：自我对弈生成数据并训练"""
        self.model.train()
        
        games_data, first_board, first_moves, first_winner = self.generate_self_play_games(
            num_games=games_per_epoch,
            save_games=True
        )
        
        # 打印第一局棋谱
        if first_board:
            title = f"自我对弈样例 - 结果: {first_winner} (共{first_moves}步)"
            print_game_board(first_board, title)
            print()
        
        if len(games_data) == 0:
            print("   警告：本轮没有生成有效训练样本")
            return {'total_loss': 0.0}
        
        random.shuffle(games_data)
        
        epoch_stats = defaultdict(float)
        batch_count = 0
        
        for start_idx in range(0, len(games_data), batch_size):
            batch = games_data[start_idx:start_idx + batch_size]
            stats = self.train_step(batch)
            
            for k, v in stats.items():
                epoch_stats[k] += v
            batch_count += 1
        
        for k in epoch_stats:
            epoch_stats[k] /= batch_count
        
        return epoch_stats

def main():
    parser = argparse.ArgumentParser(description='五子棋 Stage2 - 自我对弈训练（随机先手）')
    parser.add_argument('--new', action='store_true', 
                       help='从头开始训练（默认是继续上次的训练）')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                       help='指定检查点文件路径')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--games', '-g', type=int, default=20,
                       help='每轮对局数 (默认: 20)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("五子棋训练 Stage2 - 自我对弈（随机先手）")
    print("=" * 70)
    
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    print("\n[1/3] 初始化模型...")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )
    
    trainer = Stage2SelfPlayTrainer(model, device, lr=1e-5)
    
    if not args.new:
        checkpoint_file = None
        if args.checkpoint:
            checkpoint_file = args.checkpoint
        else:
            checkpoint_file = find_latest_checkpoint()
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            trainer.load_checkpoint(checkpoint_file)
            print("✅ 从检查点继续训练")
        else:
            # 没有stage2检查点，尝试加载stage1模型
            stage1_model = find_best_stage1_model()
            if stage1_model:
                try:
                    checkpoint = torch.load(stage1_model, map_location='cpu')
                    # 判断是checkpoint还是纯权重
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model = model.to(device)
                    print(f"✅ 成功加载 Stage1 模型: {stage1_model}")
                except Exception as e:
                    print(f"⚠️ Stage1模型加载失败: {e}, 使用随机初始化")
                    model = model.to(device)
            else:
                print("⚠️ 未找到Stage1模型，使用随机初始化")
                model = model.to(device)
    else:
        print("🆕 从头开始训练")
        # 即使--new也尝试加载stage1
        stage1_model = find_best_stage1_model()
        if stage1_model:
            try:
                checkpoint = torch.load(stage1_model, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model = model.to(device)
                print(f"✅ 加载 Stage1 模型作为初始权重: {stage1_model}")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}, 使用随机初始化")
                model = model.to(device)
        else:
            print("⚠️ 未找到Stage1模型，使用随机初始化")
            model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[2/3] 开始自我对弈训练...")
    print("-" * 90)
    
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} - 自我对弈...")
        train_stats = trainer.train_epoch(games_per_epoch=args.games, batch_size=32)
        
        print(f"\nEpoch {epoch:3d} | Loss: {train_stats.get('total_loss', 0):.4f}")
        
        trainer.save_checkpoint(epoch, f"wuziqi_stage2_checkpoint_epoch{epoch}.pth")
        trainer.scheduler.step()
    
    torch.save(model.cpu().state_dict(), "stage2/wuziqi_stage2_final.pth")
    
    print("\n[3/3] 训练完成！")
    print("=" * 70)
    print("Stage2训练完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
