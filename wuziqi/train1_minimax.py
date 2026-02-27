# train_stage1_minimax_winner.py
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
os.makedirs("stage1", exist_ok=True)

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

def find_best_stage0_model():
    """查找stage0的最佳模型（按准确率）"""
    model_files = glob.glob("stage0/wuziqi_stage0_best_*.pth")
    
    if not model_files:
        model_files = glob.glob("stage0/wuziqi_stage0_final.pth")
        if model_files:
            return model_files[0]
        return None
    
    best_model = None
    best_acc = 0
    for f in model_files:
        try:
            acc_str = f.split('best_')[1].split('%')[0]
            acc = float(acc_str)
            if acc > best_acc:
                best_acc = acc
                best_model = f
        except:
            continue
    
    if best_model:
        print(f"   找到最佳Stage0模型: {best_model} (准确率 {best_acc}%)")
        return best_model
    
    return max(model_files, key=os.path.getctime) if model_files else None

def find_latest_checkpoint():
    """查找最新的stage1检查点"""
    checkpoint_files = glob.glob("stage1/wuziqi_stage1_winner_checkpoint_epoch*.pth")
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

def save_games_to_file(games_list, filename="history_stage1_winner.json"):
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

# ============ 简化加速版 Minimax ============
def quick_evaluate(board):
    """优化版评估函数：只在棋子周围检查威胁"""
    if check_win(board, BLACK):
        return 10000
    if check_win(board, WHITE):
        return -10000
    
    score = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    # 找出所有棋子的位置
    stones = [i for i in range(BOARD_POSITIONS) if board[i] != EMPTY]
    if not stones:
        return 0
    
    # 只检查棋子周围2格内的位置（减少检查范围）
    check_positions = set()
    threat_check_positions = set()  # 专门用于威胁检查
    
    for pos in stones:
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        # 棋型检查范围（3格）
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    check_positions.add(nr * BOARD_SIZE + nc)
        
        # 威胁检查范围（2格，更快）
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    threat_check_positions.add(nr * BOARD_SIZE + nc)
    
    # 只检查棋子周围的空位是否有威胁
    black_threats = 0
    white_threats = 0
    
    for pos in threat_check_positions:
        if board[pos] != EMPTY:
            continue
        
        # 检查黑棋下这里是否能赢
        board[pos] = BLACK
        if check_win(board, BLACK):
            black_threats += 1
        board[pos] = EMPTY
        
        # 检查白棋下这里是否能赢
        board[pos] = WHITE
        if check_win(board, WHITE):
            white_threats += 1
        board[pos] = EMPTY
    
    # 威胁惩罚
    if black_threats > 0:
        score -= 5000 * black_threats
    if white_threats > 0:
        score += 5000 * white_threats
    
    # 棋型评分（只检查棋子周围）
    for i in check_positions:
        if board[i] == EMPTY:
            continue
        
        r, c = i // BOARD_SIZE, i % BOARD_SIZE
        current_player = board[i]
        
        for dr, dc in directions:
            count = 1
            # 正方向
            for step in range(1, 5):
                nr, nc = r + dr * step, c + dc * step
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    break
                if board[nr * BOARD_SIZE + nc] == current_player:
                    count += 1
                else:
                    break
            # 反方向
            for step in range(1, 5):
                nr, nc = r - dr * step, c - dc * step
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    break
                if board[nr * BOARD_SIZE + nc] == current_player:
                    count += 1
                else:
                    break
            
            # 进攻分
            multiplier = 1 if current_player == BLACK else -1
            if count >= 5:
                score += multiplier * 10000
            elif count == 4:
                score += multiplier * 1000
            elif count == 3:
                score += multiplier * 100
            elif count == 2:
                score += multiplier * 10
    
    return score

def get_noisy_minimax_move(board, player, depth=2, temperature=0.3):
    """带噪声的Minimax走法，增加多样性"""
    # 获取所有合法走法
    legals = get_nearby_moves(board, distance=2)
    if not legals:
        legals = get_legal_moves(board)
    
    if not legals:
        return None
    
    # 如果深度为0或棋盘快满了，随机选择
    if depth == 0 or len(legals) < 3:
        return random.choice(legals)
    
    maximizing = (player == BLACK)
    
    # 快速评估每个走法
    move_scores = []
    for move in legals:
        board[move] = player
        score = quick_evaluate(board)
        board[move] = EMPTY
        move_scores.append(score)
    
    # 转换为概率（温度参数控制随机性）
    scores = np.array(move_scores)
    
    # 归一化分数到合理范围，避免exp溢出
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
    scores = np.clip(scores, -10, 10)
    
    if maximizing:
        # 最大化玩家：分数越高概率越大
        probs = np.exp(scores / temperature)
    else:
        # 最小化玩家：分数越低概率越大
        probs = np.exp(-scores / temperature)
    
    probs = probs / probs.sum()
    
    # 按概率选择
    return np.random.choice(legals, p=probs)

# ============ 赢家学习 + 输家惩罚训练器（带随机性） ============

class WinnerTrainer:
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc
        }
        full_path = os.path.join("stage1", filename)
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
            self.best_acc = checkpoint.get('best_acc', 0)
            print(f"   ✅ 从 {filename} 恢复训练")
            print(f"      上次训练到 Epoch {checkpoint['epoch']}, 最佳准确率: {self.best_acc:.2%}")
            return True
        return False
    
    def generate_games(self, num_games=10, depth1=2, depth2=1, save_games=True):
        """
        让两个不同深度的Minimax对弈（带随机性）
        depth1: 第一个深度
        depth2: 第二个深度
        谁赢就学谁的走法（正奖励），谁输就惩罚谁的走法（负奖励）
        """
        training_data = []  # 每个元素: {'board': board, 'player': player, 'move': move, 'reward': reward}
        games_records = []
        win_count_depth1 = 0
        win_count_depth2 = 0
        draw_count = 0
        first_game_board = None
        first_game_moves = 0
        first_game_winner = None
        total_moves = 0
        
        for game_idx in range(num_games):
            # 随机决定谁先手
            depth1_first = random.choice([True, False])
            
            board = [EMPTY] * BOARD_POSITIONS
            current = BLACK
            game_steps = []
            moves_history = []
            
            # 第一步随机化（不一定是中心）
            center = BOARD_SIZE // 2
            if random.random() < 0.7:  # 70%概率下在中心附近随机位置
                all_moves = get_nearby_moves(board, distance=3)
                if all_moves:
                    first_move = random.choice(all_moves)
                else:
                    first_move = center * BOARD_SIZE + center
            else:
                first_move = center * BOARD_SIZE + center
            
            board[first_move] = current
            game_steps.append((board.copy(), current, first_move))
            moves_history.append((board.copy(), current, first_move))
            current = WHITE
            move_count = 1
            
            while move_count < 60:
                is_depth1_turn = (current == BLACK and depth1_first) or (current == WHITE and not depth1_first)
                
                if is_depth1_turn:
                    # 深度1的玩家，用稍高的温度增加随机性
                    move = get_noisy_minimax_move(board, current, depth=depth1, temperature=0.4)
                else:
                    # 深度2的玩家，用稍低的温度保持一定水平
                    move = get_noisy_minimax_move(board, current, depth=depth2, temperature=0.3)
                
                if move is None:
                    break
                
                board[move] = current
                game_steps.append((board.copy(), current, move))
                moves_history.append((board.copy(), current, move))
                move_count += 1
                
                if check_win(board, current):
                    break
                
                current = 3 - current
            
            # 确定胜负
            if check_win(board, BLACK):
                winner = BLACK
                winner_str = "black"
                winner_cn = "黑棋"
                if depth1_first:
                    win_count_depth1 += 1
                else:
                    win_count_depth2 += 1
            elif check_win(board, WHITE):
                winner = WHITE
                winner_str = "white"
                winner_cn = "白棋"
                if not depth1_first:
                    win_count_depth1 += 1
                else:
                    win_count_depth2 += 1
            else:
                winner = None
                winner_str = "draw"
                winner_cn = "平局"
                draw_count += 1
            
            # 分配奖励
            gamma = 0.95
            T = len(game_steps)
            
            if winner is not None:
                for t, (board_state, player, move) in enumerate(game_steps):
                    steps_to_end = T - t
                    if player == winner:
                        # 赢家的走法得正奖励
                        reward = 1.0 * (gamma ** steps_to_end)
                    else:
                        # 输家的走法得负奖励
                        reward = -0.5 * (gamma ** steps_to_end)
                    
                    training_data.append({
                        'board': board_state,
                        'player': 0 if player == BLACK else 1,
                        'move': move,
                        'reward': reward
                    })
                    total_moves += 1
            
            # 保存第一局用于打印
            if game_idx == 0:
                first_game_board = moves_history
                first_game_moves = move_count
                first_game_winner = winner_cn
            
            # 保存棋局记录
            if save_games:
                games_records.append({
                    'timestamp': datetime.now().isoformat(),
                    'epoch': self.start_epoch,
                    'game_id': game_idx,
                    'depth1_first': depth1_first,
                    'depth1': depth1,
                    'depth2': depth2,
                    'winner': winner_str,
                    'moves': [(pos_to_str(move), 'black' if player == BLACK else 'white') 
                             for _, player, move in moves_history]
                })
        
        if save_games and games_records:
            save_games_to_file(games_records)
        
        print(f"   对局结果: 深度{depth1}胜={win_count_depth1}, 深度{depth2}胜={win_count_depth2}, 平局={draw_count}")
        print(f"   收集到 {total_moves} 个训练样本")
        
        return training_data, first_game_board, first_game_moves, first_game_winner
    
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
        
        # 前向传播
        policy_logits, values = self.model.forward_with_mask(
            boards, players, legal_moves=legal_moves_list
        )
        
        policy_logits = torch.clamp(policy_logits, -20, 20)
        
        # 策略梯度损失
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 优势函数：reward - baseline (values)
        advantages = rewards - values.squeeze()
        advantages = torch.clamp(advantages, -1.0, 1.0)
        
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # 价值损失（让values拟合rewards）
        value_loss = F.mse_loss(values.squeeze(-1), rewards)
        
        # 熵正则化（鼓励探索）
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
    
    def train_epoch(self, games_per_epoch=10, depth1=2, depth2=1, batch_size=32):
        """每个epoch：让两个Minimax对弈，收集数据并训练"""
        self.model.train()
        
        training_data, first_board, first_moves, first_winner = self.generate_games(
            num_games=games_per_epoch, 
            depth1=depth1,
            depth2=depth2,
            save_games=True
        )
        
        # 打印第一局棋谱
        if first_board:
            title = f"对弈样例 - 结果: {first_winner} (深度{depth1} vs 深度{depth2}, 共{first_moves}步)"
            print_game_board(first_board, title)
            print()
        
        if len(training_data) == 0:
            print("   警告：本轮没有收集到训练数据")
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'avg_reward': 0.0
            }
        
        random.shuffle(training_data)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_avg_reward = 0.0
        batch_count = 0
        
        for start_idx in range(0, len(training_data), batch_size):
            batch = training_data[start_idx:start_idx + batch_size]
            stats = self.train_step(batch)
            
            total_loss += stats['total_loss']
            total_policy_loss += stats['policy_loss']
            total_value_loss += stats['value_loss']
            total_entropy += stats['entropy']
            total_avg_reward += stats['avg_reward']
            batch_count += 1
        
        avg_stats = {
            'total_loss': total_loss / batch_count if batch_count > 0 else 0.0,
            'policy_loss': total_policy_loss / batch_count if batch_count > 0 else 0.0,
            'value_loss': total_value_loss / batch_count if batch_count > 0 else 0.0,
            'entropy': total_entropy / batch_count if batch_count > 0 else 0.0,
            'avg_reward': total_avg_reward / batch_count if batch_count > 0 else 0.0
        }
        
        print(f"   训练统计: Loss={avg_stats['total_loss']:.4f} "
              f"(Policy={avg_stats['policy_loss']:.4f}, "
              f"Value={avg_stats['value_loss']:.4f}, "
              f"Entropy={avg_stats['entropy']:.4f})")
        print(f"   平均奖励: {avg_stats['avg_reward']:.4f}")
        
        return avg_stats

def main():
    parser = argparse.ArgumentParser(description='五子棋 Stage1 - 赢家学习 + 输家惩罚（带随机性）')
    parser.add_argument('--new', action='store_true', 
                       help='从头开始训练（默认是继续上次的训练）')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                       help='指定检查点文件路径')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--games', '-g', type=int, default=10,
                       help='每轮对局数 (默认: 10)')
    parser.add_argument('--depth1', type=int, default=2,
                       help='第一个深度 (默认: 2)')
    parser.add_argument('--depth2', type=int, default=2,
                       help='第二个深度 (默认: 2)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("五子棋训练 Stage1 - 赢家学习 + 输家惩罚（带随机性）")
    print("=" * 70)
    print(f"深度{args.depth1} vs 深度{args.depth2}")
    
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    print("\n[1/3] 初始化模型...")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )
    
    trainer = WinnerTrainer(model, device, lr=1e-4)
    
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
            stage0_model = find_best_stage0_model()
            if stage0_model:
                try:
                    checkpoint = torch.load(stage0_model, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model = model.to(device)
                    print(f"✅ 成功加载 Stage0 模型: {stage0_model}")
                except Exception as e:
                    print(f"⚠️ Stage0模型加载失败: {e}, 使用随机初始化")
                    model = model.to(device)
            else:
                print("⚠️ 未找到Stage0模型，使用随机初始化")
                model = model.to(device)
    else:
        print("🆕 从头开始训练")
        stage0_model = find_best_stage0_model()
        if stage0_model:
            try:
                checkpoint = torch.load(stage0_model, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model = model.to(device)
                print(f"✅ 加载 Stage0 模型作为初始权重: {stage0_model}")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}, 使用随机初始化")
                model = model.to(device)
        else:
            print("⚠️ 未找到Stage0模型，使用随机初始化")
            model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[2/3] 开始对弈训练...")
    print("-" * 90)
    
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} - 对弈 (深度{args.depth1} vs 深度{args.depth2})...")
        train_stats = trainer.train_epoch(
            games_per_epoch=args.games, 
            depth1=args.depth1,
            depth2=args.depth2,
            batch_size=32
        )
        
        print(f"\nEpoch {epoch:3d} | Total Loss: {train_stats['total_loss']:.4f} | Policy: {train_stats['policy_loss']:.4f} | Value: {train_stats['value_loss']:.4f} | Reward: {train_stats['avg_reward']:.4f}")
        
        trainer.save_checkpoint(epoch, f"wuziqi_stage1_winner_checkpoint_epoch{epoch}.pth")
        trainer.scheduler.step()
    
    print("\n[3/3] 训练完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
