import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 当前脚本所在目录

# generate.py
import pickle
import random
import numpy as np
from collections import defaultdict
from game import (
    check_win, get_legal_moves, get_nearby_moves, get_nearby_positions,
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS
)

def is_real_winning_move(board, player, pos):
    """检查是否是真正的获胜机会 - 落子后能直接获胜"""
    board[pos] = player
    result = check_win(board, player)
    board[pos] = EMPTY
    return result

def is_real_threat(board, player, pos):
    """检查是否是真正的威胁 - 对手落子后能直接获胜"""
    opponent = 3 - player
    board[pos] = opponent
    result = check_win(board, opponent)
    board[pos] = EMPTY
    return result

def has_four_in_a_row(board, player, pos):
    """检查落子后是否形成四子（潜在威胁）"""
    board[pos] = player
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    
    for dr, dc in directions:
        count = 1
        # 正方向
        for step in range(1, 4):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        # 反方向
        for step in range(1, 4):
            nr, nc = r - dr * step, c - dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        
        if count >= 4:
            board[pos] = EMPTY
            return True
    
    board[pos] = EMPTY
    return False

def find_real_winning_moves(board, player):
    """查找真正的获胜位置"""
    winning = []
    for pos in get_nearby_moves(board, distance=2):
        if is_real_winning_move(board, player, pos):
            winning.append(pos)
    return winning

def find_real_threats(board, player):
    """查找真正的威胁位置"""
    threats = []
    for pos in get_nearby_moves(board, distance=2):
        if is_real_threat(board, player, pos):
            threats.append(pos)
    return threats

def find_four_threats(board, player):
    """查找四子威胁（潜在威胁）"""
    fours = []
    for pos in get_nearby_moves(board, distance=2):
        if has_four_in_a_row(board, player, pos):
            fours.append(pos)
    return fours

def generate_board_with_moves(num_moves_range=(15, 30)):
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

def generate_greed_scenarios_bulk(num_scenarios=15000):
    """批量生成真正的贪婪场景（直接获胜）"""
    scenarios = []
    print(f"   目标: {num_scenarios} 个贪婪场景")
    
    while len(scenarios) < num_scenarios:
        moves_history, winner = generate_board_with_moves((15, 30))
        
        if len(moves_history) > 10:
            # 从后半部分选，更容易出现获胜机会
            min_idx = max(5, len(moves_history) // 2)
            max_idx = len(moves_history) - 2
            if max_idx <= min_idx:
                continue
                
            idx = random.randint(min_idx, max_idx)
            board, _, value = moves_history[idx]
            current = BLACK if idx % 2 == 0 else WHITE
            
            # 查找真正的获胜位置
            winning = find_real_winning_moves(board, current)
            
            if len(winning) >= 1:
                greed_label = [0.0] * BOARD_POSITIONS
                for pos in winning:
                    greed_label[pos] = 1.0
                
                action = random.choice(winning)
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    action,
                    board_value,
                    None,
                    greed_label,
                    'greed'
                ))
        
        if len(scenarios) % 100 == 0 and len(scenarios) > 0:
            print(f"      已生成 {len(scenarios)}/{num_scenarios} 贪婪场景")
    
    return scenarios[:num_scenarios]

def generate_fear_scenarios_bulk(num_scenarios=15000):
    """批量生成真正的恐惧场景（对手直接获胜）"""
    scenarios = []
    print(f"   目标: {num_scenarios} 个恐惧场景")
    
    while len(scenarios) < num_scenarios:
        moves_history, winner = generate_board_with_moves((15, 30))
        
        if len(moves_history) > 10:
            # 从后半部分选
            min_idx = max(5, len(moves_history) // 2)
            max_idx = len(moves_history) - 2
            if max_idx <= min_idx:
                continue
                
            idx = random.randint(min_idx, max_idx)
            board, _, value = moves_history[idx]
            current = BLACK if idx % 2 == 0 else WHITE
            opponent = 3 - current
            
            # 查找对手的直接威胁
            threats = find_real_threats(board, current)  # 注意：这里传current，找对手的威胁
            
            if len(threats) >= 1:
                fear_label = [0.0] * BOARD_POSITIONS
                for pos in threats:
                    fear_label[pos] = 1.0
                
                action = random.choice(threats)
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    action,
                    board_value,
                    fear_label,
                    None,
                    'fear'
                ))
        
        if len(scenarios) % 100 == 0 and len(scenarios) > 0:
            print(f"      已生成 {len(scenarios)}/{num_scenarios} 恐惧场景")
    
    return scenarios[:num_scenarios]

def generate_mixed_scenarios_bulk(num_scenarios=10000):
    """生成混合场景（既有获胜机会又有威胁）"""
    scenarios = []
    print(f"   目标: {num_scenarios} 个混合场景")
    
    while len(scenarios) < num_scenarios:
        moves_history, winner = generate_board_with_moves((20, 35))
        
        if len(moves_history) > 15:
            min_idx = max(8, len(moves_history) // 2)
            max_idx = len(moves_history) - 3
            if max_idx <= min_idx:
                continue
                
            idx = random.randint(min_idx, max_idx)
            board, _, value = moves_history[idx]
            current = BLACK if idx % 2 == 0 else WHITE
            
            winning = find_real_winning_moves(board, current)
            threats = find_real_threats(board, current)
            
            if len(winning) > 0 and len(threats) > 0:
                fear_label = [0.0] * BOARD_POSITIONS
                greed_label = [0.0] * BOARD_POSITIONS
                
                for pos in threats:
                    fear_label[pos] = 1.0
                for pos in winning:
                    greed_label[pos] = 1.0
                
                # 优先选获胜位置
                if random.random() < 0.7:
                    action = random.choice(winning)
                else:
                    action = random.choice(threats)
                
                board_value = value if value is not None else 0
                
                scenarios.append((
                    board.copy(),
                    action,
                    board_value,
                    fear_label,
                    greed_label,
                    'mixed'
                ))
        
        if len(scenarios) % 100 == 0 and len(scenarios) > 0:
            print(f"      已生成 {len(scenarios)}/{num_scenarios} 混合场景")
    
    return scenarios[:num_scenarios]

def generate_normal_scenarios_bulk(num_scenarios=20000):
    """批量生成普通场景（无直接获胜和威胁）"""
    scenarios = []
    print(f"   目标: {num_scenarios} 个普通场景")
    
    while len(scenarios) < num_scenarios:
        moves_history, winner = generate_board_with_moves((8, 20))
        
        for idx, (board, action, value) in enumerate(moves_history):
            if random.random() < 0.3 and idx > 2 and idx < len(moves_history) - 2:
                current = BLACK if idx % 2 == 0 else WHITE
                
                winning = find_real_winning_moves(board, current)
                threats = find_real_threats(board, current)
                
                if len(winning) == 0 and len(threats) == 0:
                    scenarios.append((
                        board.copy(),
                        action,
                        value if value is not None else 0,
                        None,
                        None,
                        'normal'
                    ))
        
        if len(scenarios) % 100 == 0 and len(scenarios) > 0:
            print(f"      已生成 {len(scenarios)}/{num_scenarios} 普通场景")
    
    return scenarios[:num_scenarios]

def generate_large_dataset(
    num_fear=15000,
    num_greed=15000,
    num_mixed=10000,
    num_normal=20000
):
    print("=" * 70)
    print("🚀 五子棋数据集生成器 (真正威胁版)")
    print("=" * 70)
    print(f"\n目标总量: {num_fear + num_greed + num_mixed + num_normal:,} 个样本")
    
    all_data = []
    
    print(f"\n[1/4] 生成恐惧场景（对手直接获胜）...")
    fear = generate_fear_scenarios_bulk(num_fear)
    all_data.extend(fear)
    
    print(f"\n[2/4] 生成贪婪场景（自己直接获胜）...")
    greed = generate_greed_scenarios_bulk(num_greed)
    all_data.extend(greed)
    
    print(f"\n[3/4] 生成混合场景...")
    mixed = generate_mixed_scenarios_bulk(num_mixed)
    all_data.extend(mixed)
    
    print(f"\n[4/4] 生成普通场景...")
    normal = generate_normal_scenarios_bulk(num_normal)
    all_data.extend(normal)
    
    random.shuffle(all_data)
    
    print(f"\n📊 最终数据集统计:")
    print(f"   {'='*40}")
    print(f"   总样本数: {len(all_data):,}")
    
    counts = defaultdict(int)
    fear_with_label = 0
    greed_with_label = 0
    
    for item in all_data:
        if len(item) >= 6:
            scene_type = item[5]
            counts[scene_type] += 1
            if item[3] is not None:
                fear_with_label += 1
            if item[4] is not None:
                greed_with_label += 1
    
    for stype, cnt in counts.items():
        print(f"   {stype:>10}: {cnt:6,} ({cnt/len(all_data):6.1%})")
    
    print(f"\n   恐惧标签: {fear_with_label:,} 个样本")
    print(f"   贪婪标签: {greed_with_label:,} 个样本")
    
    output_file = "wuziqi_dataset_real.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)
    
    print(f"\n✅ 数据集已保存到: {output_file}")
    print(f"   文件大小: {len(all_data) * 225 * 8 / 1024 / 1024:.1f} MB")
    
    # 显示一个样本示例
    if len(all_data) > 0:
        print(f"\n📌 样本示例:")
        sample = all_data[0]
        if len(sample) >= 6:
            board, action, value, fear, greed, stype = sample[:6]
            from game import pos_to_str
            print(f"   类型: {stype}, 动作: {pos_to_str(action)}, 价值: {value}")
            if fear is not None:
                fear_pos = [i for i, v in enumerate(fear) if v > 0]
                print(f"   恐惧位置: {[pos_to_str(p) for p in fear_pos]}")
            if greed is not None:
                greed_pos = [i for i, v in enumerate(greed) if v > 0]
                print(f"   贪婪位置: {[pos_to_str(p) for p in greed_pos]}")
    
    return all_data

if __name__ == "__main__":
    import time
    start = time.time()
    
    # 先测试小规模
    print("开始小规模测试...")
    generate_large_dataset(
        num_fear=2000,
        num_greed=2000,
        num_mixed=200,
        num_normal=2000
    )
    
    elapsed = time.time() - start
    print(f"\n⏱️ 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
