# train_stage1_vs_mcts.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import math
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
    checkpoint_files = glob.glob("stage1/wuziqi_stage1_vs_mcts_checkpoint_epoch*.pth")
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

def save_games_to_file(games_list, filename="history_stage1_vs_mcts.json"):
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

# ============ 增强版评估函数 ============

def evaluate_position_at(board, pos, player):
    """评估如果在pos下子，能形成多大的威胁"""
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    max_score = 0
    
    for dr, dc in directions:
        count = 1  # 当前下的这步棋
        
        # 正方向
        for step in range(1, 5):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        
        # 反方向
        for step in range(1, 5):
            nr, nc = r - dr * step, c - dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        
        # 根据长度给分
        if count >= 5:
            max_score = max(max_score, 10000)
        elif count == 4:
            max_score = max(max_score, 1000)
        elif count == 3:
            max_score = max(max_score, 100)
        elif count == 2:
            max_score = max(max_score, 10)
    
    return max_score

def evaluate_position(board, player):
    """增强版局面评估函数
    - 检查自己的四连（即将获胜）
    - 检查对手的三连（必须防守）
    """
    if check_win(board, player):
        return 10000
    if check_win(board, 3 - player):
        return -10000
    
    score = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    # 找出所有棋子的位置
    stones = [i for i in range(BOARD_POSITIONS) if board[i] != EMPTY]
    if not stones:
        return 0
    
    # 检查所有可能形成五子连珠的位置
    check_positions = set()
    for pos in stones:
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    check_positions.add(nr * BOARD_SIZE + nc)
    
    # 检查每个位置的棋型
    for i in check_positions:
        if board[i] == EMPTY:
            # 检查如果下在这里能形成什么
            # 1. 检查下黑棋的效果
            board[i] = BLACK
            black_score = evaluate_position_at(board, i, BLACK)
            board[i] = EMPTY
            
            # 2. 检查下白棋的效果
            board[i] = WHITE
            white_score = evaluate_position_at(board, i, WHITE)
            board[i] = EMPTY
            
            # 根据当前玩家加分
            if player == BLACK:
                score += black_score * 1.0  # 自己的进攻
                score -= white_score * 1.5  # 对手的威胁（需要防守）
            else:
                score -= black_score * 1.5  # 对手的威胁
                score += white_score * 1.0  # 自己的进攻
        else:
            # 已有棋子的位置，直接评估
            current_player = board[i]
            multiplier = 1 if current_player == player else -1
            
            for dr, dc in directions:
                count = 1
                # 正方向
                for step in range(1, 5):
                    nr, nc = (i // BOARD_SIZE) + dr * step, (i % BOARD_SIZE) + dc * step
                    if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                        break
                    if board[nr * BOARD_SIZE + nc] == current_player:
                        count += 1
                    else:
                        break
                # 反方向
                for step in range(1, 5):
                    nr, nc = (i // BOARD_SIZE) - dr * step, (i % BOARD_SIZE) - dc * step
                    if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                        break
                    if board[nr * BOARD_SIZE + nc] == current_player:
                        count += 1
                    else:
                        break
                
                if count >= 5:
                    score += multiplier * 10000
                elif count == 4:
                    # 检查是否是活四（两端无阻挡）
                    # 简化处理，都加高分
                    score += multiplier * 1000
                elif count == 3:
                    score += multiplier * 100
                elif count == 2:
                    score += multiplier * 10
    
    return score

# ============ 增强版 MCTS ============

def heuristic_simulate(board, player, max_moves=15):
    """启发式模拟：使用评估函数指导选择"""
    sim_board = board.copy()
    sim_player = player
    move_count = 0
    
    while move_count < max_moves:
        legals = get_nearby_moves(sim_board, distance=2)
        if not legals:
            legals = get_legal_moves(sim_board)
        if not legals:
            break
        
        # 用评估函数给每个走法打分
        move_scores = []
        for move in legals:
            sim_board[move] = sim_player
            score = evaluate_position(sim_board, sim_player)
            sim_board[move] = EMPTY
            move_scores.append(max(score, 0) + 1)  # 确保正数
        
        # 按分数加权随机选择
        scores = np.array(move_scores, dtype=float)
        if scores.sum() > 0:
            probs = scores / scores.sum()
            move = np.random.choice(legals, p=probs)
        else:
            move = random.choice(legals)
        
        sim_board[move] = sim_player
        
        if check_win(sim_board, sim_player):
            return 1 if sim_player == player else 0
        
        sim_player = 3 - sim_player
        move_count += 1
    
    return 0.5

class MCTSNode:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board.copy()
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = get_nearby_moves(board, distance=2)
        if not self.untried_moves:
            self.untried_moves = get_legal_moves(board)
        
        # 用评估函数排序，好的走法优先探索
        if self.untried_moves:
            move_scores = []
            for m in self.untried_moves:
                self.board[m] = self.player
                score = evaluate_position(self.board, self.player)
                self.board[m] = EMPTY
                move_scores.append(score)
            # 按分数降序排序
            sorted_moves = [m for _, m in sorted(zip(move_scores, self.untried_moves), reverse=True)]
            self.untried_moves = sorted_moves
    
    def ucb_score(self, total_visits, exploration=1.4):
        if self.visits == 0:
            return float('inf')
        win_rate = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(total_visits) / self.visits)
        return win_rate + explore
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration=1.4):
        total_visits = sum(c.visits for c in self.children)
        return max(self.children, key=lambda c: c.ucb_score(total_visits, exploration))

def mcts_search(board, player, iterations=200):
    """增强版MCTS搜索"""
    root = MCTSNode(board, player)
    
    for _ in range(iterations):
        node = root
        
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # Expansion
        if node.untried_moves:
            move = node.untried_moves.pop(0)
            new_board = node.board.copy()
            new_board[move] = node.player
            child = MCTSNode(new_board, 3 - node.player, node, move)
            node.children.append(child)
            node = child
        
        # Simulation (使用启发式模拟)
        result = heuristic_simulate(node.board, node.player, max_moves=15)
        
        # Backpropagation
        while node:
            node.visits += 1
            if node.player == player:
                node.wins += result
            else:
                node.wins += 1 - result
            node = node.parent
    
    # 选择访问次数最多的子节点
    if not root.children:
        legals = get_legal_moves(board)
        return random.choice(legals) if legals else None
    
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move

# ============ 模型 vs MCTS 训练器 ============

class VsMCTSTrainer:
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
    
    def generate_vs_mcts_games(self, num_games=5, mcts_iterations=200, save_games=True):
        """
        让模型与增强版MCTS对弈
        - 模型赢：模型的走法得正奖励
        - 模型输：模型的走法得负奖励，MCTS的走法得正奖励
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
            mcts_moves = []   # 记录MCTS的走法 (board_before, player, move)
            
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
                    # MCTS下棋
                    board_before = board.copy()
                    player_before = current
                    
                    move = mcts_search(board, current, iterations=mcts_iterations)
                    if move is None:
                        break
                    
                    mcts_moves.append({
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
                    winner_vs = "mcts"
            elif check_win(board, WHITE):
                winner = WHITE
                winner_str = "white"
                winner_cn = "白棋"
                if not model_first:
                    win_count += 1
                    winner_vs = "model"
                else:
                    loss_count += 1
                    winner_vs = "mcts"
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
                
            elif winner_vs == "mcts":
                # 模型输了：模型的走法得负奖励，MCTS的走法得正奖励
                # 模型的走法负奖励
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
                        reward = -0.5 * (gamma ** steps_to_end)
                        training_data.append({
                            'board': move_data['board'],
                            'player': move_data['player'],
                            'move': move_data['move'],
                            'reward': reward
                        })
                        total_moves += 1
                
                # MCTS的走法正奖励
                for move_data in mcts_moves:
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
                    'mcts_moves': len(mcts_moves),
                    'mcts_iterations': mcts_iterations,
                    'moves': [(pos_to_str(move), 'black' if player == BLACK else 'white') 
                             for _, player, move in moves_history]
                })
        
        if save_games and games_records:
            save_games_to_file(games_records)
        
        win_rate = win_count / num_games if num_games > 0 else 0
        print(f"   对局结果: 模型胜={win_count}, MCTS胜={loss_count}, 平局={draw_count} (胜率={win_rate:.2%})")
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
    
    def train_epoch(self, games_per_epoch=5, mcts_iterations=200, batch_size=32):
        """每个epoch：让模型与MCTS对弈，收集数据并训练"""
        self.model.train()
        
        training_data, first_board, first_moves, first_winner, win_rate = self.generate_vs_mcts_games(
            num_games=games_per_epoch, 
            mcts_iterations=mcts_iterations,
            save_games=True
        )
        
        if first_board:
            title = f"模型 vs MCTS 对弈样例 - 结果: {first_winner} (共{first_moves}步)"
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
    parser = argparse.ArgumentParser(description='五子棋 Stage1 - 模型 vs 增强版MCTS')
    parser.add_argument('--new', action='store_true', 
                       help='从头开始训练（默认是继续上次的训练）')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None,
                       help='指定检查点文件路径')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--games', '-g', type=int, default=5,
                       help='每轮对局数 (默认: 5)')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='MCTS迭代次数 (默认: 200)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("五子棋训练 Stage1 - 模型 vs 增强版MCTS")
    print("=" * 70)
    print(f"MCTS迭代次数: {args.iterations}")
    
    device = get_best_device()
    print(f"\n使用设备: {device}")
    
    print("\n[1/3] 初始化模型...")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )
    
    trainer = VsMCTSTrainer(model, device, lr=1e-4)
    
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
    
    print("\n[2/3] 开始模型 vs 增强版MCTS 训练...")
    print("-" * 90)
    
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} - 模型 vs MCTS (迭代={args.iterations})...")
        train_stats = trainer.train_epoch(
            games_per_epoch=args.games, 
            mcts_iterations=args.iterations,
            batch_size=32
        )
        
        print(f"\nEpoch {epoch:3d} | Total Loss: {train_stats['total_loss']:.4f} | Policy: {train_stats['policy_loss']:.4f} | Value: {train_stats['value_loss']:.4f} | Reward: {train_stats['avg_reward']:.4f} | 胜率: {train_stats['win_rate']:.2%}")
        
        if train_stats['win_rate'] > trainer.best_win_rate:
            trainer.best_win_rate = train_stats['win_rate']
            torch.save(model.cpu().state_dict(), f"stage1/wuziqi_stage1_vs_mcts_best_{train_stats['win_rate']:.0%}_epoch{epoch}.pth")
            model.to(device)
            print(f"          🏆 新最佳模型! 胜率={train_stats['win_rate']:.2%}")
        
        trainer.save_checkpoint(epoch, f"wuziqi_stage1_vs_mcts_checkpoint_epoch{epoch}.pth")
        trainer.scheduler.step()
    
    print("\n[3/3] 训练完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
