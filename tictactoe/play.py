# play.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from model import FearGreedModel
from game import check_win, is_full, get_legal_moves

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board):
    print("\n   " + "-" * 13)
    for i in range(3):
        print("   |", end="")
        for j in range(3):
            idx = i * 3 + j
            val = board[idx]
            
            if val == 1:
                display = "X"
            elif val == 2:
                display = "O"
            else:
                display = " "
            
            print(f" {display} |", end="")
        print("\n   " + "-" * 13)

def print_analysis(decision, player):
    print("\n" + "=" * 60)
    print(f"🤖 模型分析 (轮到 {'X' if player == 1 else 'O'})")
    print("=" * 60)
    
    value = decision['value']
    print(f"\n📈 局面价值: {value:+.4f} ", end="")
    if value > 0.3:
        print("(X优势)")
    elif value < -0.3:
        print("(O优势)")
    else:
        print("(均势)")
    
    print("\n😨 恐惧分数 (对手威胁):")
    fear = decision['fear']
    for i in range(3):
        row = ""
        for j in range(3):
            idx = i * 3 + j
            f = fear[idx]
            if f > 0.8:
                row += f" \033[91m{f:.2f}\033[0m "
            elif f > 0.5:
                row += f" \033[93m{f:.2f}\033[0m "
            elif f > 0.2:
                row += f" \033[92m{f:.2f}\033[0m "
            else:
                row += f" {f:.2f} "
        print(row)
    
    print("\n💰 贪婪分数 (获胜机会):")
    greed = decision['greed']
    for i in range(3):
        row = ""
        for j in range(3):
            idx = i * 3 + j
            g = greed[idx]
            if g > 0.8:
                row += f" \033[91m{g:.2f}\033[0m "
            elif g > 0.5:
                row += f" \033[93m{g:.2f}\033[0m "
            elif g > 0.2:
                row += f" \033[92m{g:.2f}\033[0m "
            else:
                row += f" {g:.2f} "
        print(row)
    
    print("\n📊 最终策略:")
    policy = decision['policy']
    for i in range(3):
        row = ""
        for j in range(3):
            idx = i * 3 + j
            p = policy[idx]
            if p > 0.3:
                row += f" \033[91m{p:.2f}\033[0m "
            elif p > 0.1:
                row += f" \033[93m{p:.2f}\033[0m "
            else:
                row += f" {p:.2f} "
        print(row)
    
    print(f"\n✅ 最终选择: 位置 {decision['move']}")
    print("=" * 60)

def play():
    clear_screen()
    print("=" * 70)
    print("🧠 恐惧与贪婪 AI 对战")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 使用设备: {device}")
    
    model = FearGreedModel().to(device)
    
    model_files = glob.glob("model_best_*.pth")
    if model_files:
        latest_model = max(model_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]) if 'epoch' in x else 0)
        try:
            model.load_state_dict(torch.load(latest_model, map_location=device))
            print(f"✅ 加载模型: {latest_model}")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            print("使用随机初始化模型")
    else:
        print("⚠️ 未找到训练模型，使用随机初始化")
    
    print("\n选择先后手:")
    print("   1. 我先手 (X)")
    print("   2. AI先手 (O)")
    
    while True:
        choice = input("\n请输入 (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("输入无效")
    
    human_first = (choice == '1')
    
    board = [0] * 9
    current_player = 1
    
    clear_screen()
    print("\n🎮 游戏开始!")
    print("\n初始棋盘:")
    print_board(board)
    
    while True:
        if check_win(board, 1):
            print("\n🏆 X 胜利!")
            print_board(board)
            break
        if check_win(board, 2):
            print("\n🏆 O 利!")
            print_board(board)
            break
        if is_full(board):
            print("\n🤝 平局!")
            print_board(board)
            break
        
        is_human = (current_player == 1 and human_first) or \
                   (current_player == 2 and not human_first)
        
        if is_human:
            print(f"\n👤 轮到你了 ({'X' if current_player == 1 else 'O'})")
            legals = get_legal_moves(board)
            print(f"可选位置: {legals}")
            
            while True:
                try:
                    pos = int(input("请选择位置: ").strip())
                    if pos in legals:
                        break
                    print(f"❌ 位置 {pos} 不合法")
                except ValueError:
                    print("请输入数字")
                except KeyboardInterrupt:
                    print("\n\n游戏结束")
                    return
            
            board[pos] = current_player
            print("\n当前棋盘:")
            print_board(board)
            
        else:
            print(f"\n🤖 AI 思考中...")
            
            decision = model.decide_move(board, current_player, device, debug=False)
            
            print_analysis(decision, current_player)
            
            pos = decision['move']
            board[pos] = current_player
            
            print(f"\n✅ AI 选择了位置 {pos}")
            print("\n当前棋盘:")
            print_board(board)
        
        current_player = 3 - current_player
    
    again = input("\n再玩一局？(y/n): ").strip().lower()
    if again == 'y':
        play()

if __name__ == "__main__":
    play()
