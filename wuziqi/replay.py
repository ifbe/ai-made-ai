# replay.py
import json
import argparse
import os
import sys
import tty
import termios
import random
from game import BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS, str_to_pos, pos_to_str

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def getch():
    """获取单个字符输入（不需要回车）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_game_board(moves_history, title=None):
    """
    打印一局棋的完整棋盘，用数字显示落子顺序
    黑子：白色数字在黑色背景
    白子：黑色数字在白色背景
    """
    if title:
        print(f"\n{title}")
    
    # 创建棋盘数组
    board_state = [EMPTY] * BOARD_POSITIONS
    move_numbers = [0] * BOARD_POSITIONS
    
    for step, (pos_str, player) in enumerate(moves_history):
        pos = str_to_pos(pos_str)
        if pos is None:
            continue
        board_state[pos] = BLACK if player == 'black' else WHITE
        move_numbers[pos] = step + 1
    
    # 打印列号
    print("\n    ", end="")
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
    print()
    
    # 打印上边框
    print("   ┌" + "───┬" * (BOARD_SIZE-1) + "───┐")
    
    for i in range(BOARD_SIZE):
        # 打印行号
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
        
        for j in range(BOARD_SIZE):
            idx = i * BOARD_SIZE + j
            player = board_state[idx]
            move_num = move_numbers[idx]
            
            if player == BLACK:
                # 黑子：白色数字在黑色背景
                print(f"│\033[97;40m{move_num:^3}\033[0m", end="")
            elif player == WHITE:
                # 白子：黑色数字在白色背景
                print(f"│\033[30;107m{move_num:^3}\033[0m", end="")
            else:
                # 空位
                print(f"│   ", end="")
        
        print("│")
        
        # 打印行分隔线
        if i < BOARD_SIZE - 1:
            print("   ├" + "───┼" * (BOARD_SIZE-1) + "───┤")
    
    # 打印下边框
    print("   └" + "───┴" * (BOARD_SIZE-1) + "───┘")
    
    # 打印图例
    print("\n图例: \033[97;40m 数字 \033[0m = 黑子, \033[30;107m 数字 \033[0m = 白子")

def list_games(filename="history_train1.json"):
    """列出所有保存的棋局"""
    if not os.path.exists(filename):
        print(f"❌ 文件 {filename} 不存在")
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        games = json.load(f)
    
    print(f"\n📋 共找到 {len(games)} 局棋:")
    for i, game in enumerate(games):
        timestamp = game.get('timestamp', '未知时间')
        winner = game.get('winner', '未知')
        moves = len(game.get('moves', []))
        print(f"  {i:3d}. [{timestamp}] {moves}步, 胜者: {winner}")
    
    return games

def browse_games(games):
    """交互式浏览棋局"""
    total = len(games)
    index = 0
    
    while True:
        clear_screen()
        game = games[index]
        title = f"棋局 #{index}/{total-1} - {game.get('timestamp', '未知时间')} - 胜者: {game.get('winner', '未知')}"
        print_game_board(game['moves'], title)
        
        print(f"\n📖 棋局 {index+1}/{total}")
        print("   [n]下一局  [p]上一局  [r]随机  [l]列表  [q]退出")
        print("   (直接按键，无需回车)")
        
        cmd = getch().lower()
        
        if cmd == 'n':
            index = (index + 1) % total
        elif cmd == 'p':
            index = (index - 1) % total
        elif cmd == 'r':
            index = random.randint(0, total-1)
            print(f"\n🎲 随机跳转到棋局 #{index}")
        elif cmd == 'l':
            clear_screen()
            list_games()
            print("\n按任意键继续...")
            getch()
        elif cmd == 'q':
            break

def main():
    parser = argparse.ArgumentParser(description='五子棋对局回放')
    parser.add_argument('file', nargs='?', default='history_train1.json',
                       help='棋局文件路径 (默认: history_train1.json)')
    parser.add_argument('-l', '--list', action='store_true',
                       help='列出所有棋局')
    parser.add_argument('-i', '--index', type=int,
                       help='回放指定索引的棋局')
    
    args = parser.parse_args()
    
    games = list_games(args.file)
    if not games:
        return
    
    if args.list:
        return
    
    if args.index is not None:
        if 0 <= args.index < len(games):
            game = games[args.index]
            title = f"棋局 #{args.index} - {game.get('timestamp', '未知时间')} - 胜者: {game.get('winner', '未知')}"
            print_game_board(game['moves'], title)
        else:
            print(f"❌ 索引 {args.index} 超出范围 (0-{len(games)-1})")
    else:
        # 无参数默认进入交互式浏览
        browse_games(games)

if __name__ == "__main__":
    main()
