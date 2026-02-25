import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本所在目录

import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from model import FearGreedWuziqiModel
from game import (
    check_win, is_full, get_legal_moves, get_nearby_moves,
    BLACK, WHITE, EMPTY, BOARD_SIZE, pos_to_str, str_to_pos
)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board, last_move=None, title=None):
    """打印棋盘"""
    if title:
        print(f"\n{title}")
    
    print("\n    ", end="")
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
    print()
    
    print("   ┌" + "───┬" * (BOARD_SIZE-1) + "───┐")
    
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
            print("   ├" + "───┼" * (BOARD_SIZE-1) + "───┤")
    
    print("   └" + "───┴" * (BOARD_SIZE-1) + "───┘")

def print_heatmap_grid(board, scores, title):
    """打印热力图"""
    if title:
        print(f"\n{title}")
    
    print("\n    ", end="")
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
    print()
    
    print("   ┌" + "───┬" * (BOARD_SIZE-1) + "───┐")
    
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
            print("   ├" + "───┼" * (BOARD_SIZE-1) + "───┤")
    
    print("   └" + "───┴" * (BOARD_SIZE-1) + "───┘")

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
    
    if 'policy' in decision:
        print_heatmap_grid(board, decision['policy'], "\n📊 最终策略:")
    
    print(f"\n✅ 最终选择: {pos_to_str(decision['move'])}")
    print("=" * 100)

def load_model(model_path, device):
    """加载模型"""
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    ).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ 加载模型: {model_path}")
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
    
    model = None
    stage1_files = glob.glob("wuziqi_stage1_*.pth")
    if stage1_files:
        latest = max(stage1_files)
        model = load_model(latest, device)
    
    if model is None:
        stage0_files = glob.glob("wuziqi_stage0_*.pth")
        if stage0_files:
            latest = max(stage0_files)
            model = load_model(latest, device)
    
    if model is None:
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
    
    clear_screen()
    print("\n🎮 游戏开始!")
    print_board(board)
    
    while True:
        if check_win(board, BLACK):
            print("\n🏆 黑棋(X) 胜利!")
            print_board(board, last_move)
            break
        if check_win(board, WHITE):
            print("\n🏆 白棋(O) 胜利!")
            print_board(board, last_move)
            break
        if is_full(board):
            print("\n🤝 平局!")
            print_board(board, last_move)
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
            board[pos] = current_player
            last_move = pos
            
            print(f"\n✅ AI 选择了 {pos_to_str(pos)}")
            print_board(board, last_move)
        
        current_player = 3 - current_player
    
    again = input("\n再玩一局？(y/n): ").strip().lower()
    if again == 'y':
        play()

if __name__ == "__main__":
    play()
