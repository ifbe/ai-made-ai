import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def has_four_in_row(board, player, pos):
    """检查下在pos后是否形成4连（允许更多，但不能5连）"""
    if board[pos] != EMPTY:
        return False
    
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    
    for dr, dc in directions:
        count = 1  # 当前下的子
        
        # 正方向
        for step in range(1, 5):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            idx = nr * BOARD_SIZE + nc
            if board[idx] == player:
                count += 1
            else:
                break
        
        # 反方向
        for step in range(1, 5):
            nr, nc = r - dr * step, c - dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            idx = nr * BOARD_SIZE + nc
            if board[idx] == player:
                count += 1
            else:
                break
        
        # 改为 count >= 4
        if count >= 4:
            # 还要检查是否5连
            # 简单处理：如果正好5连，也算（但实际不会，因为没下满）
            return True
    
    return False

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

def generate_winning_scenarios_bulk(num_scenarios=100):
    """生成winning场景 - 当前回合，我直接能赢（旧greed算法）"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个winning场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((15, 25))
        
        if len(moves_history) > 10:
            idx = random.randint(5, len(moves_history)-3)
            board, _, value = moves_history[idx]
            current_player = BLACK if idx % 2 == 0 else WHITE
            
            real_wins = find_real_winning_moves(board, current_player)
            
            if len(real_wins) >= 1:
                greed_label = [0.0] * BOARD_POSITIONS
                for pos in real_wins:
                    greed_label[pos] = 1.0
                
                action = random.choice(real_wins)
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    None,
                    greed_label,
                    'winning'
                ))
        
        if len(scenarios) - last_print >= 1 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} winning场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ winning场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_losing_scenarios_bulk(num_scenarios=100):
    """生成losing场景 - 当前回合，对手直接能赢（旧fear算法）"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个losing场景")
    
    last_print = 0
    attempts = 0
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        moves_history, winner = generate_board_with_moves((15, 25))
        
        if len(moves_history) > 10:
            idx = random.randint(5, len(moves_history)-3)
            board, _, value = moves_history[idx]
            current_player = BLACK if idx % 2 == 0 else WHITE
            
            real_threats = find_real_threats(board, current_player)
            
            if len(real_threats) >= 1:
                fear_label = [0.0] * BOARD_POSITIONS
                for pos in real_threats:
                    fear_label[pos] = 1.0
                
                action = random.choice(real_threats)
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    fear_label,
                    None,
                    'losing'
                ))
        
        if len(scenarios) - last_print >= 1 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} losing场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ losing场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_fear_scenarios_bulk(num_scenarios=100):
    """生成fear场景 - 对手形成4连（必败势）"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个fear场景(对手形成4连)")
    
    last_print = 0
    attempts = 0
    
    # 预定义一些5连模式（相对位置，便于平移）
    five_patterns = [
        # 横线5连
        [(0,0), (0,1), (0,2), (0,3), (0,4)],
        # 竖线5连
        [(0,0), (1,0), (2,0), (3,0), (4,0)],
        # 对角线5连
        [(0,0), (1,1), (2,2), (3,3), (4,4)],
        # 反斜线5连
        [(0,4), (1,3), (2,2), (3,1), (4,0)],
    ]
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        
        # 1. 随机选择当前玩家（黑或白）
        current_player = random.choice([BLACK, WHITE])
        opponent = 3 - current_player
        
        # 2. 随机选择一个5连模式（对手的5连）
        pattern = random.choice(five_patterns)
        
        # 3. 随机平移（确保在棋盘内）
        max_r = BOARD_SIZE - 5
        max_c = BOARD_SIZE - 5
        offset_r = random.randint(0, max_r)
        offset_c = random.randint(0, max_c)
        
        # 4. 放5个对手的子
        board = [EMPTY] * BOARD_POSITIONS
        five_positions = []
        for dr, dc in pattern:
            r = offset_r + dr
            c = offset_c + dc
            pos = r * BOARD_SIZE + c
            board[pos] = opponent
            five_positions.append(pos)
        
        # 5. 随机放3个当前玩家的子干扰（但不能放在5连位置上）
        num_player = 3
        player_positions = []
        for _ in range(num_player):
            while True:
                pos = random.randint(0, BOARD_POSITIONS-1)
                if board[pos] == EMPTY and pos not in five_positions:
                    board[pos] = current_player
                    player_positions.append(pos)
                    break
        
        # 6. 去掉对手两端的2个子（留下中间3个形成4连威胁）
        # 去掉第一个和最后一个
        fear_pos1 = five_positions[0]  # 第一个
        fear_pos2 = five_positions[4]  # 最后一个
        board[fear_pos1] = EMPTY
        board[fear_pos2] = EMPTY
        
        # 7. 这两个位置就是恐惧点（对手下这里能形成4连）
        fear_label = [0.0] * BOARD_POSITIONS
        fear_label[fear_pos1] = 1.0
        fear_label[fear_pos2] = 1.0
        
        # 8. 随机选择一个恐惧点作为动作
        action = random.choice([fear_pos1, fear_pos2])
        
        scenarios.append((
            board.copy(),
            current_player,
            action,
            0,
            fear_label,
            None,
            'fear'
        ))
        
        if len(scenarios) - last_print >= 10 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} fear场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ fear场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_greed_scenarios_bulk(num_scenarios=100):
    """生成greed场景 - 形成4连（必胜势）"""
    if num_scenarios <= 0:
        return []
    
    scenarios = []
    print(f"   目标: {num_scenarios} 个greed场景(形成4连)")
    
    last_print = 0
    attempts = 0
    
    # 预定义一些5连模式（相对位置，便于平移）
    five_patterns = [
        # 横线5连
        [(0,0), (0,1), (0,2), (0,3), (0,4)],
        # 竖线5连
        [(0,0), (1,0), (2,0), (3,0), (4,0)],
        # 对角线5连
        [(0,0), (1,1), (2,2), (3,3), (4,4)],
        # 反斜线5连
        [(0,4), (1,3), (2,2), (3,1), (4,0)],
    ]
    
    while len(scenarios) < num_scenarios:
        attempts += 1
        
        # 1. 随机选择当前玩家（黑或白）
        current_player = random.choice([BLACK, WHITE])
        opponent = 3 - current_player
        
        # 2. 随机选择一个5连模式
        pattern = random.choice(five_patterns)
        
        # 3. 随机平移（确保在棋盘内）
        max_r = BOARD_SIZE - 5
        max_c = BOARD_SIZE - 5
        offset_r = random.randint(0, max_r)
        offset_c = random.randint(0, max_c)
        
        # 4. 放5个当前玩家的子（形成5连）
        board = [EMPTY] * BOARD_POSITIONS
        five_positions = []
        for dr, dc in pattern:
            r = offset_r + dr
            c = offset_c + dc
            pos = r * BOARD_SIZE + c
            board[pos] = current_player
            five_positions.append(pos)
        
        # 5. 随机放3个对手的子干扰
        num_opponent = 3
        for _ in range(num_opponent):
            while True:
                pos = random.randint(0, BOARD_POSITIONS-1)
                if board[pos] == EMPTY:
                    board[pos] = opponent
                    break
        
        # 6. 去掉5连的两端（留下中间3个）
        # 两端的位置被移除，变成空位
        remove_pos1 = five_positions[0]  # 第一个
        remove_pos2 = five_positions[4]  # 最后一个
        board[remove_pos1] = EMPTY
        board[remove_pos2] = EMPTY
        
        # 7. 这两个空位就是贪婪点（下这里能形成4连）
        greed_label = [0.0] * BOARD_POSITIONS
        greed_label[remove_pos1] = 1.0
        greed_label[remove_pos2] = 1.0
        
        # 8. 随机选一个作为动作
        action = random.choice([remove_pos1, remove_pos2])
        
        scenarios.append((
            board.copy(),
            current_player,
            action,
            0,
            None,
            greed_label,
            'greed'
        ))
        
        if len(scenarios) - last_print >= 10 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} greed场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ greed场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_mixed_scenarios_bulk(num_scenarios=100):
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
            current_player = BLACK if idx % 2 == 0 else WHITE
            
            real_wins = find_real_winning_moves(board, current_player)
            real_threats = find_real_threats(board, current_player)
            
            if len(real_wins) > 0 and len(real_threats) > 0:
                fear_label = [0.0] * BOARD_POSITIONS
                greed_label = [0.0] * BOARD_POSITIONS
                
                for pos in real_threats:
                    fear_label[pos] = 1.0
                for pos in real_wins:
                    greed_label[pos] = 1.0
                
                action = random.choice(real_wins)
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    current_player,
                    action,
                    board_value,
                    fear_label,
                    greed_label,
                    'mixed'
                ))
        
        if len(scenarios) - last_print >= 1 and len(scenarios) > 0:
            print(f"      ✓ 已生成 {len(scenarios)}/{num_scenarios} 混合场景 (尝试次数: {attempts})")
            last_print = len(scenarios)
    
    print(f"      ✅ 混合场景完成，生成 {len(scenarios)} 个，成功率: {len(scenarios)/attempts*100:.1f}%")
    return scenarios

def generate_normal_scenarios_bulk(num_scenarios=0):
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
                current_player = BLACK if idx % 2 == 0 else WHITE
                
                real_wins = find_real_winning_moves(board, current_player)
                real_threats = find_real_threats(board, current_player)
                
                if len(real_wins) == 0 and len(real_threats) == 0:
                    scenarios.append((
                        board.copy(),
                        current_player,
                        action,
                        value if value is not None else 0,
                        None,
                        None,
                        'normal'
                    ))
        
        if len(scenarios) - last_print >= 10 and len(scenarios) > 0:
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
            if len(item) >= 7:
                _, player, _, _, _, _, scene_type = item[:7]
                counts[scene_type] += 1
                if player in [BLACK, WHITE]:
                    player_counts[player] += 1
        
        if counts:
            print(f"  场景分布:")
            total = len(data)
            for stype in ['winning', 'losing', 'greed', 'fear', 'mixed', 'normal']:
                cnt = counts.get(stype, 0)
                pct = cnt/total*100 if total > 0 else 0
                print(f"    {stype:>6}: {cnt:6} ({pct:5.1f}%)")
            
            print(f"  玩家分布:")
            print(f"    黑棋回合: {player_counts[BLACK]} ({player_counts[BLACK]/total*100:.1f}%)")
            print(f"    白棋回合: {player_counts[WHITE]} ({player_counts[WHITE]/total*100:.1f}%)")
        
        return data, counts
    return [], defaultdict(int)

def generate_large_dataset(
    num_winning=100,
    num_losing=100,
    num_greed=100,
    num_fear=100,
    num_mixed=100,
    num_normal=0,
    output_file="wuziqi_dataset_real.pkl",
    mode="continue"
):
    print("=" * 70)
    print("🚀 五子棋数据集生成器 (6场景快速版)")
    print("=" * 70)
    
    existing_data = []
    existing_counts = defaultdict(int)
    
    if mode == "continue" and os.path.exists(output_file):
        existing_data, existing_counts = load_existing_data(output_file)
    else:
        print(f"新建模式: 将生成全新数据集")
    
    # 分别计算每种场景还需要多少
    winning_needed = max(0, num_winning - existing_counts.get('winning', 0))
    losing_needed = max(0, num_losing - existing_counts.get('losing', 0))
    greed_needed = max(0, num_greed - existing_counts.get('greed', 0))
    fear_needed = max(0, num_fear - existing_counts.get('fear', 0))
    mixed_needed = max(0, num_mixed - existing_counts.get('mixed', 0))
    normal_needed = max(0, num_normal - existing_counts.get('normal', 0))
    
    total_needed = winning_needed + losing_needed + greed_needed + fear_needed + mixed_needed + normal_needed
    
    print(f"\n还需要生成:")
    print(f"   winning(直接赢): {winning_needed}")
    print(f"   losing(直接输): {losing_needed}")
    print(f"   greed(形成4连): {greed_needed}")
    print(f"   fear(对手4连): {fear_needed}")
    print(f"   mixed: {mixed_needed}")
    print(f"   normal: {normal_needed}")
    print(f"   总计: {total_needed}")
    
    if total_needed <= 0:
        print("✅ 所有场景已满足需求，无需生成")
        return existing_data
    
    all_data = existing_data if mode == "continue" else []
    
    print(f"\n[1/6] 生成winning场景(直接赢)...")
    start_time = time.time()
    winning = generate_winning_scenarios_bulk(winning_needed)
    all_data.extend(winning)
    print(f"   ⏱️ winning场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[2/6] 生成losing场景(直接输)...")
    start_time = time.time()
    losing = generate_losing_scenarios_bulk(losing_needed)
    all_data.extend(losing)
    print(f"   ⏱️ losing场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[3/6] 生成greed场景(形成4连)...")
    start_time = time.time()
    greed = generate_greed_scenarios_bulk(greed_needed)
    all_data.extend(greed)
    print(f"   ⏱️ greed场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[4/6] 生成fear场景(对手4连)...")
    start_time = time.time()
    fear = generate_fear_scenarios_bulk(fear_needed)
    all_data.extend(fear)
    print(f"   ⏱️ fear场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[5/6] 生成混合场景...")
    start_time = time.time()
    mixed = generate_mixed_scenarios_bulk(mixed_needed)
    all_data.extend(mixed)
    print(f"   ⏱️ 混合场景耗时: {time.time()-start_time:.1f}秒")
    
    print(f"\n[6/6] 生成普通场景...")
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
    
    for stype in ['winning', 'losing', 'greed', 'fear', 'mixed', 'normal']:
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
    parser = argparse.ArgumentParser(description='五子棋数据集生成器 (6场景快速版)')
    parser.add_argument('--new', action='store_true', help='重新生成数据集')
    parser.add_argument('--continue', '-c', dest='continue_mode', action='store_true', help='继续添加')
    parser.add_argument('--output', '-o', type=str, default='wuziqi_dataset_real.pkl', help='输出文件')
    parser.add_argument('--winning', type=int, default=100, help='winning场景(直接赢)')
    parser.add_argument('--losing', type=int, default=100, help='losing场景(直接输)')
    parser.add_argument('--greed', type=int, default=100, help='greed场景(形成4连)')
    parser.add_argument('--fear', type=int, default=100, help='fear场景(对手4连)')
    parser.add_argument('--mixed', type=int, default=100, help='混合场景')
    parser.add_argument('--normal', type=int, default=0, help='普通场景')
    
    args = parser.parse_args()
    mode = "new" if args.new else "continue"
    
    start = time.time()
    generate_large_dataset(
        num_winning=args.winning,
        num_losing=args.losing,
        num_greed=args.greed,
        num_fear=args.fear,
        num_mixed=args.mixed,
        num_normal=args.normal,
        output_file=args.output,
        mode=mode
    )
    elapsed = time.time() - start
    print(f"\n⏱️ 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
