import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本所在目录

# play.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
import json
import time
from datetime import datetime
from model import FearGreedWuziqiModel
from game import (
    check_win, is_full, get_legal_moves, get_nearby_moves,
    BLACK, WHITE, EMPTY, BOARD_SIZE, pos_to_str, str_to_pos
)

# 创建 stage3 目录（用于存放微调模型）
os.makedirs("stage3", exist_ok=True)

# 对局历史文件名
PLAY_HISTORY_FILE = "history_play.json"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board, last_move=None, title=None):
    """打印棋盘"""
    if title:
        print(f"\n{title}")
    
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
            val = board[idx]
            
            if val == BLACK:
                piece = "X"
            elif val == WHITE:
                piece = "O"
            else:
                piece = " "
            
            if last_move is not None and last_move == idx:
                print(f"│\033[7m{piece:^3}\033[0m", end="")
            else:
                print(f"│{piece:^3}", end="")
        
        print("│")
        
        if i < BOARD_SIZE - 1:
            print("   " + "├───" * BOARD_SIZE + "┤")
    
    print("   " + "└───" * BOARD_SIZE + "┘")

def print_heatmap_grid(board, scores, title):
    """打印热力图"""
    if title:
        print(f"\n{title}")
    
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
            score = scores[idx]
            
            if board[idx] != EMPTY:
                piece = "X" if board[idx] == BLACK else "O"
                print(f"│ \033[90m{piece}\033[0m ", end="")
            else:
                if score > 0.7:
                    print(f"│\033[91m{score:.1f}\033[0m", end="")
                elif score > 0.4:
                    print(f"│\033[93m{score:.1f}\033[0m", end="")
                elif score > 0.1:
                    print(f"│\033[92m{score:.1f}\033[0m", end="")
                else:
                    print(f"│{score:.1f}", end="")
        
        print("│")
        
        if i < BOARD_SIZE - 1:
            print("   " + "├───" * BOARD_SIZE + "┤")
    
    print("   " + "└───" * BOARD_SIZE + "┘")

def print_analysis(decision, player, board):
    """打印详细分析"""
    print("\n" + "=" * 100)
    print(f"🤖 模型分析 (轮到 {'黑棋(X)' if player == BLACK else '白棋(O)'})")
    print("=" * 100)
    
    value = decision.get('value', 0)
    print(f"\n📈 局面价值: {value:+.4f} ", end="")
    if value > 0.3:
        print("(黑棋明显优势)")
    elif value > 0.1:
        print("(黑棋略优)")
    elif value < -0.3:
        print("(白棋明显优势)")
    elif value < -0.1:
        print("(白棋略优)")
    else:
        print("(均势)")
    
    attention = decision.get('attention', 0.5)
    print(f"\n⚖️  恐惧/贪婪平衡: {attention:.3f} ", end="")
    if attention > 0.6:
        print("(偏向进攻)")
    elif attention < 0.4:
        print("(偏向防守)")
    else:
        print("(平衡)")
    
    if 'fear' in decision:
        print_heatmap_grid(board, decision['fear'], "\n😨 恐惧分数 (对手威胁):")
    
    if 'greed' in decision:
        print_heatmap_grid(board, decision['greed'], "\n💰 贪婪分数 (自己机会):")
    
    if 'normal' in decision:  # 第三个头
        print_heatmap_grid(board, decision['normal'], "\n📊 普通策略分数 (基础走法):")
    
    if 'policy' in decision:
        print_heatmap_grid(board, decision['policy'], "\n🎯 最终策略 (融合后):")
    
    print(f"\n✅ 最终选择: {pos_to_str(decision['move'])}")
    print("=" * 100)

def find_best_model():
    """查找最佳模型：优先stage3，然后stage2（取epoch最大），最后stage0（取准确率最高）"""
    # 1. 优先找stage3的微调模型（取最新的）
    stage3_files = glob.glob("stage3/wuziqi_stage3_*.pth")
    if stage3_files:
        latest = max(stage3_files, key=os.path.getctime)
        return latest, "stage3"
    
    # 2. 其次找stage2的模型（取epoch最大的，因为evaluate不准）
    stage2_files = glob.glob("stage2/wuziqi_stage2_checkpoint_epoch*.pth")
    if stage2_files:
        # 从文件名提取epoch号，取最大的
        best_model = None
        max_epoch = 0
        for f in stage2_files:
            try:
                epoch_str = f.split('epoch')[1].split('.')[0]
                epoch = int(epoch_str)
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_model = f
            except:
                continue
        if best_model:
            return best_model, f"stage2 (epoch {max_epoch})"
    
    # 3. 然后找stage0的最佳模型（按准确率）
    stage0_files = glob.glob("stage0/wuziqi_stage0_best_*.pth")
    if stage0_files:
        best_model = None
        best_acc = 0
        for f in stage0_files:
            try:
                acc_str = f.split('best_')[1].split('%')[0]
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_model = f
            except:
                continue
        if best_model:
            return best_model, f"stage0 (acc {best_acc}%)"
    
    # 4. 最后找final模型
    final_files = glob.glob("stage2/wuziqi_stage2_final.pth") + glob.glob("stage0/wuziqi_stage0_final.pth")
    if final_files:
        return final_files[0], "final"
    
    return None, None

def save_game_to_file(game_data, filename=PLAY_HISTORY_FILE):
    """保存对局到文件（追加模式）"""
    games = []
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                games = json.load(f)
        except:
            games = []
    
    games.append(game_data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(games, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 对局已保存到 {filename}")

def fine_tune_from_game(model, game_data, device, lr=1e-5):
    """用玩家对局数据微调模型"""
    # 导入需要的常量
    from game import BOARD_POSITIONS, EMPTY, BLACK, WHITE, get_nearby_moves, str_to_pos
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    moves = game_data['moves']
    winner = game_data['winner']
    
    # 转换为训练数据
    boards = []
    players = []
    actions = []
    rewards = []
    
    board = [EMPTY] * BOARD_POSITIONS
    current_player = BLACK  # 1
    
    for step, (pos_str, player_str) in enumerate(moves):
        pos = str_to_pos(pos_str)
        if pos is None:
            continue
        
        # 记录这一步
        boards.append(board.copy())
        # 转换：BLACK(1) -> 0, WHITE(2) -> 1
        players.append(0 if current_player == BLACK else 1)
        actions.append(pos)
        
        # 根据胜负计算奖励
        if winner == 'player':
            # 玩家赢了：玩家的每一步都是好的，AI的每一步都是坏的
            if (player_str == 'black' and game_data['player_color'] == 'black') or \
               (player_str == 'white' and game_data['player_color'] == 'white'):
                reward = 0.1  # 玩家的好棋
            else:
                reward = -0.1  # AI的坏棋
        elif winner == 'ai':
            # AI赢了：AI的每一步都是好的，玩家的每一步都是坏的
            if (player_str == 'black' and game_data['ai_color'] == 'black') or \
               (player_str == 'white' and game_data['ai_color'] == 'white'):
                reward = 0.1  # AI的好棋
            else:
                reward = -0.1  # 玩家的坏棋
        else:
            reward = 0.0  # 平局
        
        rewards.append(reward)
        
        # 更新棋盘
        board[pos] = current_player
        current_player = 3 - current_player
    
    # 转换为tensor
    boards_tensor = torch.tensor(boards, dtype=torch.long, device=device)
    players_tensor = torch.tensor(players, dtype=torch.long, device=device)
    actions_tensor = torch.tensor(actions, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
    
    # 训练几步
    model.train()
    total_loss = 0
    
    for i in range(0, len(boards), 16):  # batch_size=16
        batch_boards = boards_tensor[i:i+16]
        batch_players = players_tensor[i:i+16]
        batch_actions = actions_tensor[i:i+16]
        batch_rewards = rewards_tensor[i:i+16]
        
        # 获取附近位置作为合法移动
        legal_moves_list = []
        for b in range(len(batch_boards)):
            board_state = batch_boards[b].cpu().numpy()
            nearby = get_nearby_moves(board_state, distance=2)
            if not nearby:
                center = BOARD_SIZE // 2
                nearby = [center * BOARD_SIZE + center]
            legal_moves_list.append(nearby)
        
        policy_logits, values = model.forward_with_mask(
            batch_boards, batch_players, legal_moves=legal_moves_list
        )
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
        
        advantages = batch_rewards - values.squeeze()
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(-1), batch_rewards)
        
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    model.eval()
    print(f"   🤖 微调完成，平均损失: {total_loss/len(boards):.4f}")
    
    # 保存微调后的模型到 stage3 目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"stage3/wuziqi_stage3_{timestamp}.pth"
    torch.save(model.cpu().state_dict(), model_path)
    model.to(device)
    print(f"   💾 微调模型已保存: {model_path}")
    
    return model

def load_model(model_path, device):
    """加载模型 - 支持普通权重和checkpoint"""
    model = FearGreedWuziqiModel().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 判断是checkpoint还是纯权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 这是checkpoint文件
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载checkpoint模型: {model_path}")
            if 'epoch' in checkpoint:
                print(f"   epoch: {checkpoint['epoch']}")
            if 'best_win_rate' in checkpoint:
                print(f"   最佳胜率: {checkpoint['best_win_rate']:.2%}")
        else:
            # 这是纯权重文件
            model.load_state_dict(checkpoint)
            print(f"✅ 加载权重模型: {model_path}")
        
        return model
    except Exception as e:
        print(f"⚠️ 加载失败: {e}")
        return None

def play():
    clear_screen()
    print("=" * 100)
    print("🎮 五子棋恐惧与贪婪 AI")
    print("=" * 100)
    print("\n📝 输入格式: 两个字符，如 00 表示左上角")
    print("   列: 0-9 a-e")
    print("   行: 0-9 a-e")
    print("\n棋子: X = 黑棋, O = 白棋")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 使用设备: {device}")
    
    # 查找最佳模型（优先stage3）
    model_path, model_type = find_best_model()
    if model_path:
        model = load_model(model_path, device)
        print(f"   使用 {model_type} 模型")
    else:
        print("⚠️ 未找到训练模型，使用随机初始化")
        model = FearGreedWuziqiModel(
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256
        ).to(device)
    
    print("\n选择先后手:")
    print("   1. 我先手 (黑棋 X)")
    print("   2. AI先手 (白棋 O)")
    
    while True:
        choice = input("\n请输入 (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("输入无效")
    
    human_first = (choice == '1')
    
    board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
    current_player = BLACK
    last_move = None
    moves_history = []  # 记录对局历史
    
    clear_screen()
    print("\n🎮 游戏开始!")
    print_board(board)
    
    while True:
        if check_win(board, BLACK):
            winner = "black"
            winner_str = "黑棋"
            break
        if check_win(board, WHITE):
            winner = "white"
            winner_str = "白棋"
            break
        if is_full(board):
            winner = "draw"
            winner_str = "平局"
            break
        
        is_human = (current_player == BLACK and human_first) or \
                   (current_player == WHITE and not human_first)
        
        if is_human:
            print(f"\n👤 轮到你了 ({'黑棋(X)' if current_player == BLACK else '白棋(O)'})")
            
            legals = get_legal_moves(board)
            nearby = get_nearby_moves(board, distance=2)
            print(f"推荐落子区域: {[pos_to_str(p) for p in nearby[:5]]}")
            
            while True:
                try:
                    pos_str = input("请选择位置 (如 00): ").strip()
                    pos = str_to_pos(pos_str)
                    if pos is None:
                        print("❌ 格式错误")
                    elif pos not in legals:
                        print(f"❌ 位置 {pos_str} 不合法")
                    else:
                        break
                except KeyboardInterrupt:
                    print("\n\n游戏结束")
                    return
            
            board[pos] = current_player
            moves_history.append((pos_str, 'black' if current_player == BLACK else 'white'))
            last_move = pos
            
            clear_screen()
            print_board(board, last_move)
            
        else:
            print(f"\n🤖 AI 思考中...")
            
            decision = model.decide_move_fast(board, current_player, device, debug=False)
            
            clear_screen()
            print_board(board, last_move)
            print_analysis(decision, current_player, board)
            
            pos = decision['move']
            pos_str = pos_to_str(pos)
            board[pos] = current_player
            moves_history.append((pos_str, 'black' if current_player == BLACK else 'white'))
            last_move = pos
            
            print(f"\n✅ AI 选择了 {pos_str}")
            print_board(board, last_move)
        
        current_player = 3 - current_player
    
    # 游戏结束，显示结果
    print(f"\n🏆 游戏结束! 胜者: {winner_str}")
    print_board(board, last_move)
    
    # 保存对局历史
    game_data = {
        'timestamp': datetime.now().isoformat(),
        'player_color': 'black' if human_first else 'white',
        'ai_color': 'white' if human_first else 'black',
        'winner': 'player' if (winner == 'black' and human_first) or (winner == 'white' and not human_first) else
                 'ai' if (winner == 'black' and not human_first) or (winner == 'white' and human_first) else
                 'draw',
        'moves': moves_history
    }
    
    save_game_to_file(game_data)
    
    # 询问是否继续
    while True:
        again = input("\n再玩一局？(y/n/t): ").strip().lower()
        if again in ['y', 'n', 't']:
            break
        print("输入无效，请输入 y(是)/n(否)/t(训练)")
    
    if again == 'y':
        play()
    elif again == 't':
        print(f"\n🔄 正在用本局数据微调模型...")
        model = fine_tune_from_game(model, game_data, device)
        # 微调后自动开始新一局
        play()
    else:
        print("\n👋 感谢游玩，再见！")

if __name__ == "__main__":
    play()
