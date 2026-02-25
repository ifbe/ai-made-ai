# showdata.py
import pickle
import argparse
import random
import numpy as np
from collections import defaultdict
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS,
    pos_to_str
)

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_heatmap_grid(board, scores, title, highlight_move=None):
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
            score = scores[idx] if scores is not None else 0
            
            if board[idx] != EMPTY:
                piece = "X" if board[idx] == BLACK else "O"
                if highlight_move is not None and highlight_move == idx:
                    print(f"│\033[7m \033[90m{piece}\033[0m\033[0m ", end="")
                else:
                    print(f"│ \033[90m{piece}\033[0m ", end="")
            else:
                if highlight_move is not None and highlight_move == idx:
                    if score > 0.7:
                        print(f"│\033[7m\033[91m{score:.1f}\033[0m", end="")
                    elif score > 0.4:
                        print(f"│\033[7m\033[93m{score:.1f}\033[0m", end="")
                    elif score > 0.1:
                        print(f"│\033[7m\033[92m{score:.1f}\033[0m", end="")
                    else:
                        print(f"│\033[7m{score:.1f}\033[0m", end="")
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

def print_sample_details(item, index):
    """打印样本详情"""
    if len(item) >= 6:
        board, action, value, fear_label, greed_label, scene_type = item[:6]
    else:
        board, action, value = item[:3]
        scene_type = 'unknown'
        fear_label = greed_label = None
    
    print("\n" + "=" * 100)
    print(f"📌 样本 #{index}")
    print("=" * 100)
    
    print(f"\n📋 基本信息:")
    print(f"   场景类型: {scene_type}")
    print(f"   正确动作: {pos_to_str(action)}")
    print(f"   局面价值: {value}")
    
    black_count = sum(1 for x in board if x == BLACK)
    white_count = sum(1 for x in board if x == WHITE)
    print(f"   棋子数量: 黑棋(X)={black_count}, 白棋(O)={white_count}")
    
    print_board(board, action, "\n🎯 棋盘:")
    
    if fear_label is not None and any(fear_label):
        fear_array = np.array(fear_label)
        print_heatmap_grid(board, fear_array, "\n😨 恐惧标签:", action)
    
    if greed_label is not None and any(greed_label):
        greed_array = np.array(greed_label)
        print_heatmap_grid(board, greed_array, "\n💰 贪婪标签:", action)
    
    print("=" * 100)

def print_stats(data):
    """打印统计信息"""
    total = len(data)
    print("\n" + "=" * 100)
    print("📊 数据集统计")
    print("=" * 100)
    
    print(f"\n总样本数: {total:,}")
    
    counts = defaultdict(int)
    values = []
    fear_counts = 0
    greed_counts = 0
    
    for item in data:
        if len(item) >= 6:
            scene_type = item[5]
            counts[scene_type] += 1
        else:
            counts['unknown'] += 1
        
        if len(item) >= 4 and item[3] is not None:
            fear_counts += 1
        if len(item) >= 5 and item[4] is not None:
            greed_counts += 1
        if len(item) >= 3 and item[2] is not None:
            values.append(item[2])
    
    print(f"\n📌 场景分布:")
    for stype, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {stype:>10}: {cnt:6,} ({cnt/total:6.1%})")
    
    print(f"\n🏷️  标签统计:")
    print(f"   恐惧标签: {fear_counts:6,} ({fear_counts/total:6.1%})")
    print(f"   贪婪标签: {greed_counts:6,} ({greed_counts/total:6.1%})")
    
    if values:
        print(f"\n💰 价值分布:")
        print(f"   黑胜 (1): {values.count(1):6,} ({values.count(1)/len(values):6.1%})")
        print(f"   白胜 (-1): {values.count(-1):6,} ({values.count(-1)/len(values):6.1%})")
        print(f"   平局 (0): {values.count(0):6,} ({values.count(0)/len(values):6.1%})")

def browse_data(data):
    """交互式浏览"""
    total = len(data)
    index = 0
    
    while True:
        clear_screen()
        print_sample_details(data[index], index)
        
        print(f"\n📖 样本 {index+1}/{total}")
        print("   [n]下一个  [p]上一个  [j]跳转  [q]退出")
        
        cmd = input("\n请输入命令: ").strip().lower()
        
        if cmd == 'n' or cmd == '':
            index = (index + 1) % total
        elif cmd == 'p':
            index = (index - 1) % total
        elif cmd == 'j':
            try:
                new_idx = int(input("输入样本索引: ").strip())
                if 0 <= new_idx < total:
                    index = new_idx
                else:
                    print(f"索引必须在 0-{total-1} 之间")
                    input("按Enter继续...")
            except:
                print("输入无效")
                input("按Enter继续...")
        elif cmd == 'q':
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='wuziqi_dataset_real.pkl')
    parser.add_argument('-s', '--stats', action='store_true', help='统计信息')
    parser.add_argument('-i', '--index', type=int, help='显示指定索引')
    parser.add_argument('-n', '--num', type=int, default=3, help='显示前N个')
    parser.add_argument('-b', '--browse', action='store_true', help='浏览模式')
    parser.add_argument('-t', '--type', help='按类型筛选')
    
    args = parser.parse_args()
    
    try:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
        print(f"\n✅ 加载 {args.file}, 共 {len(data):,} 条")
    except Exception as e:
        print(f"❌ 无法加载 {args.file}: {e}")
        return
    
    if args.type:
        filtered = [item for item in data if len(item) >= 6 and item[5] == args.type]
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
                input("\n按Enter继续...")

if __name__ == "__main__":
    main()
