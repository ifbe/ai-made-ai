

# test_model.py
import torch
import torch.nn.functional as F
import pickle
import random
import argparse
import numpy as np
import sys
import tty
import termios
import os
import time
from model import FearGreedWuziqiModel
from game import (
    check_win, get_legal_moves, get_nearby_moves,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS, pos_to_str
)

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

def print_board_with_labels(board, fear_scores=None, greed_scores=None, normal_scores=None, policy=None, action=None):
    """打印棋盘，可选择显示各个头的分数"""
    
    # 打印列号
    print("\n   ", end="")
    for i in range(BOARD_SIZE):
        if i < 10:
            print(f"│ {i} ", end="")
        else:
            print(f"│ {chr(ord('a') + (i-10))} ", end="")
    print("│")
    
    print("   " + "├───" * BOARD_SIZE + "┤")
    
    for i in range(BOARD_SIZE):
        # 打印行号
        if i < 10:
            print(f" {i} ", end="")
        else:
            print(f" {chr(ord('a') + (i-10))} ", end="")
        
        for j in range(BOARD_SIZE):
            idx = i * BOARD_SIZE + j
            val = board[idx]
            
            if val != EMPTY:
                # 有棋子的位置
                piece = "X" if val == BLACK else "O"
                if action is not None and action == idx:
                    print(f"│\033[7m{piece:^3}\033[0m", end="")
                else:
                    print(f"│{piece:^3}", end="")
            else:
                # 空位，可以显示分数
                display = "   "
                color = ""
                
                # 优先显示策略分数（如果有）
                if policy is not None:
                    score = policy[idx]
                    if score > 0.01:
                        display = f"{score:.1f}"
                        if len(display) < 3:
                            display = display.ljust(3)
                        if score > 0.5:
                            color = "\033[91m"  # 红色
                        elif score > 0.1:
                            color = "\033[93m"  # 黄色
                
                if action is not None and action == idx:
                    print(f"│\033[7m{color}{display:^3}\033[0m", end="")
                else:
                    if color:
                        print(f"│{color}{display:^3}\033[0m", end="")
                    else:
                        print(f"│{display:^3}", end="")
        
        print("│")
        
        if i < BOARD_SIZE - 1:
            print("   " + "├───" * BOARD_SIZE + "┤")
    
    print("   " + "└───" * BOARD_SIZE + "┘")

def print_heatmap(board, scores, title):
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
            score = scores[idx] if scores is not None else 0
            
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

def test_sample(model, sample, device):
    """测试单个样本并返回结果"""
    # 解析样本 - 没有旧格式，直接7字段
    board, player, action, value, fear_label, greed_label, scene_type = sample[:7]
    
    # 确保player是整数（已经是1或2）
    if isinstance(player, (list, tuple, np.ndarray)):
        player = player[0] if len(player) > 0 else BLACK
    player = int(player) if player is not None else BLACK
    
    result = {
        'board': board,
        'player': player,
        'action': action,
        'value': value,
        'fear_label': fear_label,
        'greed_label': greed_label,
        'scene_type': scene_type
    }
    
    with torch.no_grad():
        board_tensor = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
        
        # 模型内部需要 0/1，但数据存的是 1/2
        # BLACK=1 -> 0, WHITE=2 -> 1
        player_tensor = torch.tensor([0 if player == BLACK else 1], device=device)
        
        details = model.forward_with_mask(board_tensor, player_tensor, return_details=True)
        
        # 获取各个头的输出
        result['fear_scores'] = details['fear_scores'][0].cpu().numpy()
        result['greed_scores'] = details['greed_scores'][0].cpu().numpy()
        result['normal_scores'] = details['normal_scores'][0].cpu().numpy()
        result['policy_logits'] = details['policy'][0].cpu().numpy()
        result['policy_probs'] = F.softmax(torch.tensor(result['policy_logits']), dim=-1).numpy()
        result['value_pred'] = details['value'][0].item()
        
        # 找出模型选择的动作
        legals = get_legal_moves(board)
        nearby = get_nearby_moves(board, distance=2)
        nearby_probs = [(pos, result['policy_probs'][pos]) for pos in nearby if pos in legals]
        if nearby_probs:
            result['model_action'] = max(nearby_probs, key=lambda x: x[1])[0]
            result['model_prob'] = result['policy_probs'][result['model_action']]
        else:
            result['model_action'] = None
            result['model_prob'] = 0
    
    return result

def display_sample(result, index, total):
    """显示样本结果"""
    clear_screen()
    
    print("=" * 100)
    print(f"📌 样本 #{index}/{total-1}")
    print("=" * 100)
    
    print(f"\n📋 基本信息:")
    print(f"   场景类型: {result['scene_type']}")
    print(f"   当前玩家: {'黑棋(X)' if result['player'] == BLACK else '白棋(O)'}")
    print(f"   正确动作: {pos_to_str(result['action'])}")
    print(f"   局面价值: {result['value']}")
    print(f"   预测价值: {result['value_pred']:.4f}")
    
    # 打印原始棋盘
    print("\n📋 原始棋盘 (高亮为正确动作):")
    print_board_with_labels(result['board'], action=result['action'])
    
    # 如果有标签，打印标签
    if result['fear_label'] is not None:
        print("\n📋 原始恐惧标签:")
        print_heatmap(result['board'], result['fear_label'], "")
    
    if result['greed_label'] is not None:
        print("\n📋 原始贪婪标签:")
        print_heatmap(result['board'], result['greed_label'], "")
    
    # 打印三个头的输出
    print("\n😨 恐惧头输出:")
    print_heatmap(result['board'], result['fear_scores'], "")
    
    print("\n💰 贪婪头输出:")
    print_heatmap(result['board'], result['greed_scores'], "")
    
    print("\n📊 普通头输出:")
    print_heatmap(result['board'], result['normal_scores'], "")
    
    # 打印最终策略
    print("\n🎯 最终策略 (概率):")
    print_board_with_labels(result['board'], policy=result['policy_probs'], action=result['action'])
    
    # 模型选择
    if result['model_action'] is not None:
        print(f"\n✅ 模型选择: {pos_to_str(result['model_action'])} (概率: {result['model_prob']:.3f})")
        print(f"   正确动作: {pos_to_str(result['action'])}")
        if result['model_action'] == result['action']:
            print("   🎉 模型选择正确!")
        else:
            print("   ❌ 模型选择错误")
    
    # 分析恐惧头
    print("\n🔍 恐惧头分析:")
    fear_positions = [(i, result['fear_scores'][i]) for i in range(len(result['fear_scores'])) 
                     if result['fear_scores'][i] > 0.1]
    fear_positions.sort(key=lambda x: x[1], reverse=True)
    if fear_positions:
        for pos, score in fear_positions[:5]:
            threat = "是" if result['fear_label'] is not None and result['fear_label'][pos] > 0 else "否"
            print(f"   位置 {pos_to_str(pos)}: 分数={score:.3f} (标签中是否是威胁: {threat})")
    else:
        print("   无显著恐惧位置")
    
    # 分析贪婪头
    print("\n💰 贪婪头分析:")
    greed_positions = [(i, result['greed_scores'][i]) for i in range(len(result['greed_scores'])) 
                      if result['greed_scores'][i] > 0.1]
    greed_positions.sort(key=lambda x: x[1], reverse=True)
    if greed_positions:
        for pos, score in greed_positions[:5]:
            is_greed = result['greed_label'] is not None and result['greed_label'][pos] > 0 if result['greed_label'] is not None else False
            print(f"   位置 {pos_to_str(pos)}: 分数={score:.3f} (标签中是否是机会: {is_greed})")
    else:
        print("   无显著贪婪位置")

def browse_samples(model, data, device):
    """交互式浏览样本"""
    total = len(data)
    index = 0
    
    while True:
        sample = data[index]
        result = test_sample(model, sample, device)
        display_sample(result, index, total)
        
        print(f"\n📖 样本 {index+1}/{total}")
        print("   [n]下一个  [p]上一个  [r]随机  [j]跳转  [q]退出")
        print("   (直接按键，无需回车)")
        
        cmd = getch().lower()
        
        if cmd == 'n':
            index = (index + 1) % total
        elif cmd == 'p':
            index = (index - 1) % total
        elif cmd == 'r':
            # 随机跳转
            index = random.randint(0, total-1)
            print(f"\n🎲 随机跳转到样本 #{index}")
            # 短暂暂停让用户看到提示
            time.sleep(0.5)
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
    parser = argparse.ArgumentParser(description='交互式测试模型对场景的反应')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--data', '-d', type=str, default='wuziqi_dataset_real.pkl',
                       help='数据集文件路径')
    parser.add_argument('--index', '-i', type=int, default=None,
                       help='起始样本索引 (默认0)')
    parser.add_argument('--cpu', action='store_true',
                       help='强制使用CPU')
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu')))
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = FearGreedWuziqiModel(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    ).to(device)
    
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    model.eval()
    
    # 加载数据集
    print(f"\n加载数据集: {args.data}")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    print(f"数据集共有 {len(data)} 个样本")
    
    # 设置起始索引
    start_index = args.index if args.index is not None else 0
    if start_index < 0 or start_index >= len(data):
        print(f"起始索引 {start_index} 超出范围，使用0")
        start_index = 0
    
    # 进入交互式浏览
    browse_samples(model, data, device)

if __name__ == "__main__":
    main()
