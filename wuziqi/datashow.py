# showdata.py
import pickle
import argparse
import random
import numpy as np
import sys
import tty
import termios
from collections import defaultdict
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS,
    pos_to_str
)

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_board_with_labels(board, player, fear_label=None, greed_label=None, action=None):
    """打印棋盘，用彩色字母显示恐惧/贪婪位置"""
    
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
            
            if val != EMPTY:
                piece = "X" if val == BLACK else "O"
                if action is not None and action == idx:
                    print(f"│\033[7m{piece:^3}\033[0m", end="")
                else:
                    print(f"│{piece:^3}", end="")
            else:
                is_fear = fear_label is not None and fear_label[idx] > 0
                is_greed = greed_label is not None and greed_label[idx] > 0
                
                if is_fear and is_greed:
                    display = "F+G"
                    color = "\033[95m"
                elif is_fear:
                    display = " F "
                    color = "\033[91m"
                elif is_greed:
                    display = " G "
                    color = "\033[92m"
                else:
                    display = "   "
                    color = ""
                
                if action is not None and action == idx:
                    if color:
                        print(f"│\033[7m{color}{display}\033[0m", end="")
                    else:
                        print(f"│\033[7m{display}\033[0m", end="")
                else:
                    if color:
                        print(f"│{color}{display}\033[0m", end="")
                    else:
                        print(f"│{display}", end="")
        
        print("│")
        
        if i < BOARD_SIZE - 1:
            print("   " + "├───" * BOARD_SIZE + "┤")
    
    print("   " + "└───" * BOARD_SIZE + "┘")

def print_sample_details(item, index):
    """打印单个样本的详细信息"""
    board, player, action, value, fear_label, greed_label, scene_type = item[:7]
    
    print("\n" + "=" * 100)
    print(f"📌 样本 #{index}")
    print("=" * 100)
    
    print(f"\n📋 基本信息:")
    print(f"   场景类型: {scene_type}")
    print(f"   当前玩家: {'黑棋(X)' if player == BLACK else '白棋(O)'}")
    print(f"   正确动作: {pos_to_str(action)}")
    print(f"   局面价值: {value}")
    
    black_count = sum(1 for x in board if x == BLACK)
    white_count = sum(1 for x in board if x == WHITE)
    print(f"   棋子数量: 黑棋(X)={black_count}, 白棋(O)={white_count}")
    
    print(f"\n🎯 棋盘 (红色F=恐惧, 绿色G=贪婪, 紫色F+G=两者):")
    print_board_with_labels(board, player, fear_label, greed_label, action)
    
    if fear_label is not None:
        fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
        if fear_positions:
            print(f"\n😨 恐惧位置 (红色F): {[pos_to_str(p) for p in fear_positions]}")
    
    if greed_label is not None:
        greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
        if greed_positions:
            print(f"\n💰 贪婪位置 (绿色G): {[pos_to_str(p) for p in greed_positions]}")
    
    print("=" * 100)

def print_stats(data):
    """打印数据集统计信息"""
    total = len(data)
    print("\n" + "=" * 100)
    print("📊 数据集统计")
    print("=" * 100)
    
    print(f"\n总样本数: {total:,}")
    
    counts = defaultdict(int)
    values = []
    fear_counts = 0
    greed_counts = 0
    player_counts = {BLACK: 0, WHITE: 0}
    
    for item in data:
        board, player, action, value, fear_label, greed_label, scene_type = item[:7]
        
        counts[scene_type] += 1
        player_counts[player] += 1
        
        if fear_label is not None:
            fear_counts += 1
        if greed_label is not None:
            greed_counts += 1
        if value is not None:
            values.append(value)
    
    print(f"\n📌 场景分布:")
    for stype, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {stype:>10}: {cnt:6,} ({cnt/total:6.1%})")
    
    print(f"\n👥 玩家分布:")
    print(f"   黑棋回合: {player_counts[BLACK]:6,} ({player_counts[BLACK]/total:6.1%})")
    print(f"   白棋回合: {player_counts[WHITE]:6,} ({player_counts[WHITE]/total:6.1%})")
    
    print(f"\n🏷️  标签统计:")
    print(f"   恐惧标签: {fear_counts:6,} ({fear_counts/total:6.1%})")
    print(f"   贪婪标签: {greed_counts:6,} ({greed_counts/total:6.1%})")
    
    if values:
        print(f"\n💰 价值分布:")
        print(f"   黑胜 (1): {values.count(1):6,} ({values.count(1)/len(values):6.1%})")
        print(f"   白胜 (-1): {values.count(-1):6,} ({values.count(-1)/len(values):6.1%})")
        print(f"   平局 (0): {values.count(0):6,} ({values.count(0)/len(values):6.1%})")

def browse_data(data):
    """交互式浏览 - 单键命令"""
    total = len(data)
    index = 0
    
    while True:
        clear_screen()
        print_sample_details(data[index], index)
        
        print(f"\n📖 样本 {index+1}/{total}")
        print("   [n]下一个  [p]上一个  [r]随机  [j]跳转  [q]退出   (直接按键，无需回车)")
        
        cmd = getch().lower()
        
        if cmd == 'n':
            index = (index + 1) % total
        elif cmd == 'p':
            index = (index - 1) % total
        elif cmd == 'r':
            # 随机跳转
            index = random.randint(0, total-1)
            print(f"\n🎲 随机跳转到样本 #{index}")
        elif cmd == 'j':
            print("\n\033[K输入样本索引: ", end="", flush=True)
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            try:
                new_idx = int(sys.stdin.readline().strip())
                if 0 <= new_idx < total:
                    index = new_idx
                else:
                    print(f"索引必须在 0-{total-1} 之间")
                    print("按任意键继续...", end="", flush=True)
                    tty.setraw(sys.stdin.fileno())
                    sys.stdin.read(1)
            except:
                print("输入无效")
                print("按任意键继续...", end="", flush=True)
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            
            tty.setraw(sys.stdin.fileno())
            
        elif cmd == 'q':
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='wuziqi_dataset_real.pkl')
    parser.add_argument('-s', '--stats', action='store_true')
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-n', '--num', type=int, default=3)
    parser.add_argument('-b', '--browse', action='store_true')
    parser.add_argument('-t', '--type')
    
    args = parser.parse_args()
    
    try:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
        print(f"\n✅ 加载 {args.file}, 共 {len(data):,} 条")
    except Exception as e:
        print(f"❌ 无法加载 {args.file}: {e}")
        return
    
    if args.type:
        filtered = [item for item in data if item[6] == args.type]
        print(f"筛选后: {len(filtered)} 条")
        data = filtered
    
    if args.stats:
        print_stats(data)
    elif args.index is not None:
        if 0 <= args.index < len(data):
            print_sample_details(data[args.index], args.index)
        else:
            print(f"索引超出范围")
    elif args.browse:
        browse_data(data)
    else:
        print_stats(data)
        print(f"\n显示前 {args.num} 个样本:")
        for i in range(min(args.num, len(data))):
            print_sample_details(data[i], i)
            if i < min(args.num, len(data)) - 1:
                print("\n按任意键继续...", end="", flush=True)
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setraw(fd)
                sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    main()
