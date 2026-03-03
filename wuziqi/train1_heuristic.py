# train_stage1_vs_heuristic.py
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
    checkpoint_files = glob.glob("stage1/wuziqi_stage1_heuristic_checkpoint_epoch*.pth")
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

def save_games_to_file(games_list, filename="history_stage1_heuristic.json"):
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

# ============ 启发式 AI ============

# 棋型评分表
PATTERNS = {
    'win5': 1000000,
    'live4': 100000,
    'rush4': 5000,
    'live3': 5000,
    'sleep3': 500,
    'live2': 500,
    'sleep2': 50,
}

def evaluate_direction(board, pos, player, dr, dc):
    """评估一个方向上的棋型"""
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    count = 1
    left_empty = 0
    right_empty = 0
    left_blocked = False
    right_blocked = False
    
    # 正方向
    for step in range(1, 5):
        nr, nc = r + dr * step, c + dc * step
        if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
            right_blocked = True
            break
        idx = nr * BOARD_SIZE + nc
        if board[idx] == player:
            count += 1
        elif board[idx] == EMPTY:
            right_empty += 1
            break
        else:
            right_blocked = True
            break
    
    # 负方向
    for step in range(1, 5):
        nr, nc = r - dr * step, c - dc * step
        if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
            left_blocked = True
            break
        idx = nr * BOARD_SIZE + nc
        if board[idx] == player:
            count += 1
        elif board[idx] == EMPTY:
            left_empty += 1
            break
        else:
            left_blocked = True
            break
    
    total_empty = left_empty + right_empty
    
    if count >= 5:
        return PATTERNS['win5']
    elif count == 4:
        if not left_blocked and not right_blocked and total_empty >= 2:
            return PATTERNS['live4']
        elif total_empty >= 1:
            return PATTERNS['rush4']
    elif count == 3:
        if not left_blocked and not right_blocked and total_empty >= 2:
            return PATTERNS['live3']
        elif total_empty >= 1:
            return PATTERNS['sleep3']
    elif count == 2:
        if not left_blocked and not right_blocked and total_empty >= 3:
            return PATTERNS['live2']
        elif total_empty >= 2:
            return PATTERNS['sleep2']
    
    return 0

def evaluate_move(board, move, player):
    """评估一个位置的分数"""
    if board[move] != EMPTY:
        return -float('inf')
    
    score = 0
    directions = [(1,0), (0,1), (1,1), (1,-1)]
    
    # 自己的得分
    board[move] = player
    for dr, dc in directions:
        score += evaluate_direction(board, move, player, dr, dc)
    board[move] = EMPTY
    
    # 对手的得分（防守分）
    opponent = 3 - player
    board[move] = opponent
    opp_score = 0
    for dr, dc in directions:
        opp_score += evaluate_direction(board, move, opponent, dr, dc)
    board[move] = EMPTY
    
    # 综合得分：进攻 + 防守
    return score * 1.2 + opp_score * 0.8

def heuristic_move(board, player):
    """启发式走法"""
    # 1. 必赢
    for pos in get_nearby_moves(board, distance=2):
        board[pos] = player
        if check_win(board, player):
            board[pos] = EMPTY
            return pos
        board[pos] = EMPTY
    
    # 2. 必防
    opponent = 3 - player
    for pos in get_nearby_moves(board, distance=2):
        board[pos] = opponent
        if check_win(board, opponent):
            board[pos] = EMPTY
            return pos
        board[pos] = EMPTY
    
    # 3. 打分
    candidates = get_nearby_moves(board, distance=2)
    if not candidates:
        center = BOARD_SIZE // 2
        return center * BOARD_SIZE + center
    
    # 给每个候选位置打分
    scores = [evaluate_move(board, m, player) for m in candidates]
    best_idx = np.argmax(scores)
    return candidates[best_idx]

# ============ 模型 vs 启发式AI 训练器 ============

class VsHeuristicTrainer:
    def __init__(self, model, device, lr=1e-4, start_epoch=1):
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
            self.best_win_rate = checkpoint.get('best_win_rate', 0)
            print(f"   ✅ 从 {filename} 恢复训练")
            print(f"      上次训练到 Epoch {checkpoint['epoch']}, 最佳胜率: {self.best_win_rate:.2%}")
            return True
        return False
    
    def generate_vs_heuristic_games(self, num_games=10, save_games=True):
        """
        让模型与启发式AI对弈
        - 模型赢：模型的走法得正奖励
        - 模型输：模型的走法得负奖励，启发式AI的走法得正奖励
        """
        training_data = []
        games_records = []
        win_count = 0
        loss_count = 0
        draw_count = 0
        first_game_board = None
        first_game_moves = 0
        first_game_winner = None
        total_moves = 0
        
        for game_idx in range(num_games):
            # 随机决定谁先手
            model_first = random.choice([True, False])
            
            board = [EMPTY] * BOARD_POSITIONS
            current = BLACK
            game_steps = []
            moves_history = []
            model_moves = []  # 记录模型的走法 (board_before, player, move)
            heuristic_moves = []   # 记录启发式AI的走法 (board_before, player, move)
            
            # 第一步随机化
            center = BOARD_SIZE // 2
            if random.random() < 0.5:
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
                is_model_turn = (current == BLACK and model_first) or (current == WHITE and not model_first)
                
                if is_model_turn:
                    # 模型下棋
                    board_before = board.copy()
                    player_before = current
                    
                    with torch.no_grad():
                        board_tensor = torch.tensor(board, dtype=torch.long, device=self.device).unsqueeze(0)
                        player_tensor = torch.tensor([0 if current == BLACK else 1], device=self.device)
                        policy_logits, _ = self.model.forward_with_mask(board_tensor, player_tensor)
                        
                        temperature = 0.3
                        probs = F.softmax(policy_logits[0] / temperature, dim=-1).cpu().numpy()
                        
                        legals = get_legal_moves(board)
                        if not legals:
                            break
                        
                        valid_probs = probs[legals]
                        if valid_probs.sum() > 0:
                            valid_probs = valid_probs / valid_probs.sum()
                            move = np.random.choice(legals, p=valid_probs)
                        else:
                            move = random.choice(legals)
                    
                    model_moves.append({
                        'board': board_before,
                        'player': 0 if player_before == BLACK else 1,
                        'move': move
                    })
                    
                else:
                    # 启发式AI下棋
                    board_before = board.copy()
                    player_before = current
                    
                    move = heuristic_move(board, current)
                    if move is None:
                        break
                    
                    heuristic_moves.append({
                        'board': board_before,
                        'player': 0 if player_before == BLACK else 1,
                        'move': move
                    })
                
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
                if model_first:
                    win_count += 1
                    winner_vs = "model"
                else:
                    loss_count += 1
                    winner_vs = "heuristic"
            elif check_win(board, WHITE):
                winner = WHITE
                winner_str = "white"
                winner_cn = "白棋"
                if not model_first:
                    win_count += 1
                    winner_vs = "model"
                else:
                    loss_count += 1
                    winner_vs = "heuristic"
            else:
                winner = None
                winner_str = "draw"
                winner_cn = "平局"
                winner_vs = "draw"
                draw_count += 1
            
            # 分配奖励
            gamma = 0.95
            T = len(game_steps)
            
            if winner_vs == "model":
                # 模型赢了：模型的走法得正奖励
                for move_data in model_moves:
                    step_idx = None
                    for t, (_, player, move) in enumerate(game_steps):
                        if (move == move_data['move'] and 
                            ((player == BLACK and move_data['player'] == 0) or 
                             (player == WHITE and move_data['player'] == 1))):
                            step_idx = t
                            break
                    
                    if step_idx is not None:
                        steps_to_end = T - step_idx
                        reward = 1.0 * (gamma ** steps_to_end)
                        training_data.append({
                            'board': move_data['board'],
                            'player': move_data['player'],
                            'move': move_data['move'],
                            'reward': reward
                        })
                        total_moves += 1
                
            elif winner_vs == "heuristic":
                # 启发式AI的走法正奖励
                for move_data in heuristic_moves:
                    step_idx = None
                    for t, (_, player, move) in enumerate(game_steps):
                        if (move == move_data['move'] and 
                            ((player == BLACK and move_data['player'] == 0) or 
                             (player == WHITE and move_data['player'] == 1))):
                            step_idx = t
                            break
                    
                    if step_idx is not None:
                        steps_to_end = T - step_idx
                        reward = 1.0 * (gamma ** steps_to_end)
                        training_data.append({
                            'board': move_data['board'],
                            'player': move_data['player'],
                            'move': move_data['move'],
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
                    'model_first': model_first,
                    'winner': winner_vs,
                    'model_moves': len(model_moves),
                    'heuristic_moves': len(heuristic_moves),
                    'moves': [(pos_to_str(move), 'black' if player == BLACK else 'white') 
                             for _, player, move in moves_history]
                })
        
        if save_games and games_records:
            save_games_to_file(games_records)
        
        win_rate = win_count / num_games if num_games > 0 else 0
        print(f"   对局结果: 模型胜={win_count}, 启发式AI胜={loss_count}, 平局={draw_count} (胜率={win_rate:.2%})")
        print(f"   收集到 {total_moves} 个训练样本")
        
        return training_data, first_game_board, first_game_moves, first_game_winner, win_rate
    
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
        
        # 价值损失
        value_loss = F.mse_loss(values.squeeze(-1), rewards)
        
        # 熵正则化
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
    
    def train_epoch(self, games_per_epoch=10, batch_size=32):
        """每个epoch：让模型与启发式AI对弈，收集数据并训练"""
        self.model.train()
        
        training_data, first_board, first_moves, first_winner, win_rate = self.generate_vs_heuristic_games(
            num_games=games_per_epoch,
            save_games=True
        )
        
        if first_board:
            title = f"模型 vs 启发式AI 对弈样例 - 结果: {first_winner} (共{first_moves}步)"
            print_game_board(first_board, title)
            print()
        
        if len(training_data) == 0:
            print("   警告：本轮没有收集到训练数据")
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'avg_reward': 0.0,
                'win_rate': win_rate
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
            'avg_reward': total_avg_reward / batch_count if batch_count > 0 else 0.0,
            'win_rate': win_rate
        }
        
        print(f"   训练统计: Loss={avg_stats['total_loss']:.4f} "
              f"(Policy={avg_stats['policy_loss']:.4f}, "
              f"Value={avg_stats['value_loss']:.4f}, "
              f"Entropy={avg_stats['entropy']:.4f})")
        print(f"   平均奖励: {avg_stats['avg_reward']:.4f}")
        
        return avg_stats

def main():
    parser = argparse.ArgumentParser(description='五子棋 Stage1 - 模型 vs 启发式AI')
    parser.add_argument('--new', action='store_true', 
                       help='从头开始训练（默认是继续上次的训练）')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                       help='指定检查点文件路径')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--games', '-g', type=int, default=100,
                       help='每轮对局数 (默认: 10)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("五子棋训练 Stage1 - 模型 vs 启发式AI")
    print("=" * 70)
    
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    print("\n[1/3] 初始化模型...")
    model = FearGreedWuziqiModel(
        state_dim=64,      # 棋盘状态维度
        pos_dim=32,        # 位置编码维度
        nhead=8,           # 注意力头数
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    trainer = VsHeuristicTrainer(model, device, lr=1e-4)
    
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
    
    print("\n[2/3] 开始模型 vs 启发式AI 训练...")
    print("-" * 90)
    
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} - 模型 vs 启发式AI...")
        train_stats = trainer.train_epoch(
            games_per_epoch=args.games,
            batch_size=32
        )
        
        print(f"\nEpoch {epoch:3d} | Total Loss: {train_stats['total_loss']:.4f} | Policy: {train_stats['policy_loss']:.4f} | Value: {train_stats['value_loss']:.4f} | Reward: {train_stats['avg_reward']:.4f} | 胜率: {train_stats['win_rate']:.2%}")
        
        if train_stats['win_rate'] > trainer.best_win_rate:
            trainer.best_win_rate = train_stats['win_rate']
            torch.save(model.cpu().state_dict(), f"stage1/wuziqi_stage1_heuristic_best_{train_stats['win_rate']:.0%}_epoch{epoch}.pth")
            model.to(device)
            print(f"          🏆 新最佳模型! 胜率={train_stats['win_rate']:.2%}")
        
        trainer.save_checkpoint(epoch, f"wuziqi_stage1_heuristic_checkpoint_epoch{epoch}.pth")
        trainer.scheduler.step()
    
    print("\n[3/3] 训练完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
