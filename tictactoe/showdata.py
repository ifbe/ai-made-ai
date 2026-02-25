# showdata.py
import pickle
import argparse

def print_board(board, highlight=None):
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
                display = str(idx)
            
            if highlight and idx in highlight:
                print(f" \033[91m{display}\033[0m |", end="")
            else:
                print(f" {display} |", end="")
        print("\n   " + "-" * 13)

def get_legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def print_stats(data):
    total = len(data)
    print(f"\n📊 数据集统计")
    print("=" * 50)
    print(f"总样本数: {total}")
    
    counts = {}
    for item in data:
        scene_type = item[5] if len(item) >= 6 else 'unknown'
        counts[scene_type] = counts.get(scene_type, 0) + 1
    
    for stype, cnt in counts.items():
        print(f"  {stype}: {cnt} ({cnt/total:.1%})")
    
    values = [d[2] for d in data if d[2] is not None]
    if values:
        print(f"\n价值分布:")
        print(f"  先手胜 (1): {values.count(1)} ({values.count(1)/len(values):.1%})")
        print(f"  后手胜 (-1): {values.count(-1)} ({values.count(-1)/len(values):.1%})")
        print(f"  平局 (0): {values.count(0)} ({values.count(0)/len(values):.1%})")

def print_sample(data, index):
    if index < 0 or index >= len(data):
        print(f"索引 {index} 超出范围")
        return
    
    item = data[index]
    print(f"\n📌 样本 #{index}")
    print("=" * 50)
    
    board, action, value = item[:3]
    scene_type = item[5] if len(item) >= 6 else 'unknown'
    fear_label = item[3] if len(item) >= 4 else None
    greed_label = item[4] if len(item) >= 5 else None
    
    print(f"场景类型: {scene_type}")
    print(f"正确动作: {action}")
    print(f"价值: {value}")
    print("\n棋盘:")
    print_board(board)
    
    if fear_label is not None:
        fear_pos = [i for i, v in enumerate(fear_label) if v > 0]
        print(f"恐惧标签: {fear_pos}")
        print_board(board, highlight=fear_pos)
    
    if greed_label is not None:
        greed_pos = [i for i, v in enumerate(greed_label) if v > 0]
        print(f"贪婪标签: {greed_pos}")
        print_board(board, highlight=greed_pos)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='fear_greed_dataset.pkl')
    parser.add_argument('-i', '--index', type=int, help='查看指定索引')
    parser.add_argument('-n', '--num', type=int, default=5, help='查看前N个')
    parser.add_argument('-s', '--stats', action='store_true', help='统计信息')
    
    args = parser.parse_args()
    
    try:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 加载 {args.file}, 共 {len(data)} 条")
    except:
        print(f"❌ 无法加载 {args.file}")
        return
    
    if args.stats:
        print_stats(data)
    elif args.index is not None:
        print_sample(data, args.index)
    else:
        print_stats(data)
        print(f"\n显示前 {args.num} 个样本:")
        for i in range(min(args.num, len(data))):
            print_sample(data, i)

if __name__ == "__main__":
    main()
