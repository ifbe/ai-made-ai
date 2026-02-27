import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本所在目录

# generate.py
# generate.py
import pickle
import random
import numpy as np
import argparse
import os
import time
from collections import defaultdict
from game import (
    check_win, get_legal_moves, get_nearby_moves, get_nearby_positions,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS
)

# ============ 核心检测函数 ============

def is_real_winning_move(board, player, pos):
    """严格检查：下这里是否能直接获胜"""
    if board[pos] != EMPTY:
        return False
    board[pos] = player
    result = check_win(board, player)
    board[pos] = EMPTY
    return result

def is_real_threat(board, player, pos):
    """严格检查：对手下这里是否能直接获胜"""
    opponent = 3 - player
    if board[pos] != EMPTY:
        return False
    board[pos] = opponent
    result = check_win(board, opponent)
    board[pos] = EMPTY
    return result

def find_real_winning_moves(board, player):
    """查找所有真正能赢的位置"""
    winning = []
    for pos in get_nearby_moves(board, distance=2):
        if is_real_winning_move(board, player, pos):
            winning.append(pos)
    return winning

def find_real_threats(board, player):
    """查找所有对手真正能赢的位置"""
    threats = []
    for pos in get_nearby_moves(board, distance=2):
        if is_real_threat(board, player, pos):
            threats.append(pos)
    return threats

# ============ 棋盘生成 ============

def generate_board_with_moves(num_moves_range=(15, 25)):
    """生成一个随机棋盘状态"""
    board = [EMPTY] * BOARD_POSITIONS
    num_moves = random.randint(*num_moves_range)
    moves_history = []
    
    center = BOARD_SIZE // 2
    center_pos = center * BOARD_SIZE + center
    board[center_pos] = BLACK
    moves_history.append((board.copy(), center_pos, None))
    current = WHITE
    
    for step in range(1, num_moves):
        nearby = get_nearby_moves(board, distance=2)
        if not nearby:
            break
        
        pos = random.choice(nearby)
        board[pos] = current
        moves_history.append((board.copy(), pos, None))
        
        if check_win(board, current):
            value = 1 if current == BLACK else -1
            for i in range(len(moves_history)):
                moves_history[i] = (moves_history[i][0], moves_history[i][1], value)
            return moves_history, current
        
        current = 3 - current
    
    for i in range(len(moves_history)):
        moves_history[i] = (moves_history[i][0], moves_history[i][1], 0)
    return moves_history, None

# ============ 场景生成 ============

def generate_fear_scenarios_bulk(num_scenarios=15000):
    """生成恐惧场景 - 只标记对手真正能赢的位置"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个恐惧场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((15, 25))
        
        if len(moves_history) > 10:
            idx = random.randint(5, len(moves_history)-3)
            board, _, value = moves_history[idx]
            current_player = BLACK if idx % 2 == 0 else WHITE  # 1或2
            
            # 严格查找对手能赢的位置
            real_threats = find_real_threats(board, current_player)
            
            # 只有确实有威胁才生成
            if len(real_threats) >= 1:
                fear_label = [0.0] * BOARD_POSITIONS
                for pos in real_threats:
                    fear_label[pos] = 1.0
                
                action = random.choice(real_threats)
                board_value = value if value is not None else 0
                
                # 新格式: board, player, action, value, fear_label, greed_label, scene_type
                # player直接存 1 或 2
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    fear_label,
                    None,
                    'fear'
                ))
        
        if len(scenarios) - last_print >= 100 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} 恐惧场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ 恐惧场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_greed_scenarios_bulk(num_scenarios=15000):
    """生成贪婪场景 - 只标记自己能真正赢的位置"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个贪婪场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((15, 25))
        
        if len(moves_history) > 10:
            idx = random.randint(5, len(moves_history)-3)
            board, _, value = moves_history[idx]
            current_player = BLACK if idx % 2 == 0 else WHITE  # 1或2
            
            # 严格查找自己能赢的位置
            real_wins = find_real_winning_moves(board, current_player)
            
            if len(real_wins) >= 1:
                greed_label = [0.0] * BOARD_POSITIONS
                for pos in real_wins:
                    greed_label[pos] = 1.0
                
                action = random.choice(real_wins)
                board_value = value if value is not None else 0
                
                # 新格式: board, player, action, value, fear_label, greed_label, scene_type
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    None,
                    greed_label,
                    'greed'
                ))
        
        if len(scenarios) - last_print >= 100 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} 贪婪场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ 贪婪场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_mixed_scenarios_bulk(num_scenarios=10000):
    """生成混合场景 - 同时有自己能赢和对手能赢的位置"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个混合场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((20, 30))
        
        if len(moves_history) > 15:
            idx = random.randint(8, len(moves_history)-5)
            board, _, value = moves_history[idx]
            current_player = BLACK if idx % 2 == 0 else WHITE  # 1或2
            
            real_wins = find_real_winning_moves(board, current_player)
            real_threats = find_real_threats(board, current_player)
            
            if len(real_wins) > 0 and len(real_threats) > 0:
                fear_label = [0.0] * BOARD_POSITIONS
                greed_label = [0.0] * BOARD_POSITIONS
                
                for pos in real_threats:
                    fear_label[pos] = 1.0
                for pos in real_wins:
                    greed_label[pos] = 1.0
                
                # 优先选能赢的位置
                action = random.choice(real_wins) if random.random() < 0.7 else random.choice(real_threats)
                board_value = value if value is not None else 0
                
                # 新格式: board, player, action, value, fear_label, greed_label, scene_type
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    fear_label,
                    greed_label,
                    'mixed'
                ))
        
        if len(scenarios) - last_print >= 100 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} 混合场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ 混合场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_normal_scenarios_bulk(num_scenarios=20000):
    """生成普通场景 - 没有任何直接获胜位置"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个普通场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((8, 20))
        
        for idx, (board, action, value) in enumerate(moves_history):
            if random.random() < 0.3 and idx > 2 and idx < len(moves_history) - 2:
                current_player = BLACK if idx % 2 == 0 else WHITE  # 1或2
                
                # 确保没有任何直接获胜位置
                real_wins = find_real_winning_moves(board, current_player)
                real_threats = find_real_threats(board, current_player)
                
                if len(real_wins) == 0 and len(real_threats) == 0:
                    # 新格式: board, player, action, value, fear_label, greed_label, scene_type
                    scenarios.append((
                        board.copy(),
                        current_player,
                        action,
                        value if value is not None else 0,
                        None,
                        None,
                        'normal'
                    ))
        
        if len(scenarios) - last_print >= 100 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} 普通场景")
            last_print = len(scenarios)
    
    return scenarios

def load_existing_data(filename):
    """加载已有的数据集并打印详细统计"""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        print(f"加载已有数据: {len(data)} 条")
        
        counts = defaultdict(int)
        player_counts = {BLACK: 0, WHITE: 0}
        
        for item in data:
            # 新格式: board, player, action, value, fear_label, greed_label, scene_type
            if len(item) >= 7:
                _, player, _, _, _, _, scene_type = item[:7]
                counts[scene_type] += 1
                if player in [BLACK, WHITE]:
                    player_counts[player] += 1
        
        if counts:
            print(f"  场景分布:")
            total = len(data)
            for stype in ['fear', 'greed', 'mixed', 'normal']:
                cnt = counts.get(stype, 0)
                pct = cnt/total*100 if total > 0 else 0
                print(f"    {stype:>6}: {cnt:6} ({pct:5.1f}%)")
            
            print(f"  玩家分布:")
            print(f"    黑棋回合: {player_counts[BLACK]} ({player_counts[BLACK]/total*100:.1f}%)")
            print(f"    白棋回合: {player_counts[WHITE]} ({player_counts[WHITE]/total*100:.1f}%)")
        
        return data, counts
    return [], defaultdict(int)

def generate_large_dataset(
    num_fear=15000,
    num_greed=15000,
    num_mixed=10000,
    num_normal=20000,
    output_file="wuziqi_dataset_real.pkl",
    mode="continue"
):
    print("=" * 70)
    print("🚀 五子棋数据集生成器")
    print("=" * 70)
    
    existing_data = []
    existing_counts = defaultdict(int)
    
    if mode == "continue" and os.path.exists(output_file):
        existing_data, existing_counts = load_existing_data(output_file)
    else:
        print(f"新建模式: 将生成全新数据集")
    
    # 分别计算每种场景还需要多少
    fear_needed = max(0, num_fear - existing_counts.get('fear', 0))
    greed_needed = max(0, num_greed - existing_counts.get('greed', 0))
    mixed_needed = max(0, num_mixed - existing_counts.get('mixed', 0))
    normal_needed = max(0, num_normal - existing_counts.get('normal', 0))
    
    total_needed = fear_needed + greed_needed + mixed_needed + normal_needed
    
    print(f"\n还需要生成:")
    print(f"   恐惧: {fear_needed}")
    print(f"   贪婪: {greed_needed}")
    print(f"   混合: {mixed_needed}")
    print(f"   普通: {normal_needed}")
    print(f"   总计: {total_needed}")
    
    if total_needed <= 0:
        print("✅ 所有场景已满足需求，无需生成")
        return existing_data
    
    all_data = existing_data if mode == "continue" else []
    
    print(f"\n[1/4] 生成恐惧场景...")
    start_time = time.time()
    fear = generate_fear_scenarios_bulk(fear_needed)
    all_data.extend(fear)
    print(f"   ⏱️ 恐惧场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[2/4] 生成贪婪场景...")
    start_time = time.time()
    greed = generate_greed_scenarios_bulk(greed_needed)
    all_data.extend(greed)
    print(f"   ⏱️ 贪婪场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[3/4] 生成混合场景...")
    start_time = time.time()
    mixed = generate_mixed_scenarios_bulk(mixed_needed)
    all_data.extend(mixed)
    print(f"   ⏱️ 混合场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[4/4] 生成普通场景...")
    start_time = time.time()
    normal = generate_normal_scenarios_bulk(normal_needed)
    all_data.extend(normal)
    print(f"   ⏱️ 普通场景耗时: {time.time()-start_time:.1f}秒")
    
    random.shuffle(all_data)
    
    print(f"\n📊 最终数据集统计:")
    print(f"   {'='*40}")
    print(f"   总样本数: {len(all_data):,}")
    
    final_counts = defaultdict(int)
    player_counts = {BLACK: 0, WHITE: 0}
    fear_with_label = 0
    greed_with_label = 0
    
    for item in all_data:
        board, player, action, value, fear_label, greed_label, scene_type = item[:7]
        final_counts[scene_type] += 1
        if player in [BLACK, WHITE]:
            player_counts[player] += 1
        if fear_label is not None:
            fear_with_label += 1
        if greed_label is not None:
            greed_with_label += 1
    
    for stype in ['fear', 'greed', 'mixed', 'normal']:
        cnt = final_counts.get(stype, 0)
        pct = cnt/len(all_data)*100 if len(all_data) > 0 else 0
        print(f"   {stype:>6}: {cnt:6} ({pct:5.1f}%)")
    
    print(f"\n  玩家分布:")
    print(f"    黑棋回合: {player_counts[BLACK]} ({player_counts[BLACK]/len(all_data)*100:.1f}%)")
    print(f"    白棋回合: {player_counts[WHITE]} ({player_counts[WHITE]/len(all_data)*100:.1f}%)")
    
    print(f"\n   恐惧标签: {fear_with_label:,} 个样本")
    print(f"   贪婪标签: {greed_with_label:,} 个样本")
    
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)
    
    print(f"\n✅ 数据集已保存到: {output_file}")
    print(f"   文件大小: {len(all_data) * 225 * 8 / 1024 / 1024:.1f} MB")
    
    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='五子棋数据集生成器')
    parser.add_argument('--new', action='store_true', help='重新生成数据集')
    parser.add_argument('--continue', '-c', dest='continue_mode', action='store_true', help='继续添加')
    parser.add_argument('--output', '-o', type=str, default='wuziqi_dataset_real.pkl', help='输出文件')
    parser.add_argument('--fear', type=int, default=100, help='恐惧场景目标')
    parser.add_argument('--greed', type=int, default=100, help='贪婪场景目标')
    parser.add_argument('--mixed', type=int, default=100, help='混合场景目标')
    parser.add_argument('--normal', type=int, default=100, help='普通场景目标')
    
    args = parser.parse_args()
    mode = "new" if args.new else "continue"
    
    start = time.time()
    generate_large_dataset(
        num_fear=args.fear,
        num_greed=args.greed,
        num_mixed=args.mixed,
        num_normal=args.normal,
        output_file=args.output,
        mode=mode
    )
    elapsed = time.time() - start
    print(f"\n⏱️ 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
