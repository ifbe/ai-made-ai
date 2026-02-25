# generate.py
import pickle
import random
from game import generate_expert_game, check_win, get_legal_moves, is_winning_move, is_threat_move

def analyze_board_threats(board, player):
    """分析对手下一步能赢的所有位置"""
    opponent = 3 - player
    threats = []
    for pos in get_legal_moves(board):
        board[pos] = opponent
        if check_win(board, opponent):
            threats.append(pos)
        board[pos] = 0
    return threats

def analyze_winning_moves(board, player):
    """分析自己下一步能赢的所有位置"""
    winning = []
    for pos in get_legal_moves(board):
        board[pos] = player
        if check_win(board, player):
            winning.append(pos)
        board[pos] = 0
    return winning

# ============ 基础恐惧场景（最简单的对手两连）============

def generate_basic_threat_scenarios(num_scenarios=5000):
    """基础威胁场景 - 最简单的对手两连，不加干扰"""
    scenarios = []
    
    patterns = [
        # 行
        ([0,1], 2), ([1,2], 0),
        ([3,4], 5), ([4,5], 3),
        ([6,7], 8), ([7,8], 6),
        # 列
        ([0,3], 6), ([3,6], 0),
        ([1,4], 7), ([4,7], 1),
        ([2,5], 8), ([5,8], 2),
        # 对角线
        ([0,4], 8), ([4,8], 0),
        ([2,4], 6), ([4,6], 2),
    ]
    
    for threat_moves, threat_pos in patterns:
        for _ in range(num_scenarios // len(patterns) + 5):
            board = [0] * 9
            # 只放威胁，不加干扰
            for pos in threat_moves:
                board[pos] = 1  # X
            
            # 验证
            board[threat_pos] = 1
            if check_win(board, 1):
                board[threat_pos] = 0
                fear_label = [0.0] * 9
                fear_label[threat_pos] = 1.0
                scenarios.append((
                    board.copy(),
                    threat_pos,
                    -1,
                    fear_label,
                    None,
                    'basic_threat'
                ))
            
            if len(scenarios) >= num_scenarios:
                break
        if len(scenarios) >= num_scenarios:
            break
    
    print(f"   生成 {len(scenarios)} 个基础威胁场景")
    return scenarios

# ============ 单威胁场景（带干扰）============

def generate_single_threat_scenarios(num_scenarios=3000):
    """单威胁场景 - 一个威胁位置，带干扰棋子"""
    scenarios = []
    
    patterns = [
        ([0,1], 2), ([1,2], 0),
        ([3,4], 5), ([4,5], 3),
        ([6,7], 8), ([7,8], 6),
        ([0,3], 6), ([3,6], 0),
        ([1,4], 7), ([4,7], 1),
        ([2,5], 8), ([5,8], 2),
        ([0,4], 8), ([4,8], 0),
        ([2,4], 6), ([4,6], 2),
    ]
    
    for threat_moves, threat_pos in patterns:
        for _ in range(num_scenarios // len(patterns) + 1):
            board = [0] * 9
            
            # 设置威胁
            for pos in threat_moves:
                board[pos] = 1  # X
            
            # 添加干扰
            empty = [p for p in range(9) if board[p] == 0 and p != threat_pos]
            random.shuffle(empty)
            num_o = random.randint(1, 3)
            for i in range(min(num_o, len(empty))):
                board[empty[i]] = 2
            
            # 验证威胁
            board[threat_pos] = 1
            is_threat = check_win(board, 1)
            board[threat_pos] = 0
            
            if is_threat:
                fear_label = [0.0] * 9
                fear_label[threat_pos] = 1.0
                
                scenarios.append((
                    board.copy(),
                    threat_pos,
                    -1,
                    fear_label,
                    None,
                    'single_threat'
                ))
            
            if len(scenarios) >= num_scenarios:
                break
        if len(scenarios) >= num_scenarios:
            break
    
    print(f"   生成 {len(scenarios)} 个单威胁场景")
    return scenarios

# ============ 双威胁场景 ============

def generate_double_threat_scenarios(num_scenarios=2000):
    """双威胁场景 - 两个威胁位置"""
    scenarios = []
    
    patterns = [
        # X在0和2，威胁1和4
        {
            'x_positions': [0, 2],
            'threats': [1, 4],
        },
        # X在0和6，威胁3和7
        {
            'x_positions': [0, 6],
            'threats': [3, 7],
        },
        # X在2和8，威胁5和7
        {
            'x_positions': [2, 8],
            'threats': [5, 7],
        },
        # X在0和8，威胁4
        {
            'x_positions': [0, 8],
            'threats': [4],
        },
        # X在2和6，威胁4
        {
            'x_positions': [2, 6],
            'threats': [4],
        },
    ]
    
    for pattern in patterns:
        for _ in range(num_scenarios // len(patterns) + 1):
            board = [0] * 9
            
            # 设置X
            for pos in pattern['x_positions']:
                board[pos] = 1
            
            # 添加O干扰
            empty = [p for p in range(9) if board[p] == 0]
            random.shuffle(empty)
            num_o = random.randint(1, 3)
            for i in range(min(num_o, len(empty))):
                board[empty[i]] = 2
            
            # 验证实际威胁
            actual_threats = []
            for pos in get_legal_moves(board):
                board[pos] = 1
                if check_win(board, 1):
                    actual_threats.append(pos)
                board[pos] = 0
            
            if len(actual_threats) >= 2:
                fear_label = [0.0] * 9
                for t in actual_threats:
                    fear_label[t] = 1.0
                
                action = random.choice(actual_threats)
                
                scenarios.append((
                    board.copy(),
                    action,
                    -1,
                    fear_label,
                    None,
                    'double_threat'
                ))
            
            if len(scenarios) >= num_scenarios:
                break
        if len(scenarios) >= num_scenarios:
            break
    
    print(f"   生成 {len(scenarios)} 个双威胁场景")
    return scenarios

# ============ 三威胁场景 ============

def generate_triple_threat_scenarios(num_scenarios=1000):
    """三威胁场景 - 三个或更多威胁位置"""
    scenarios = []
    
    patterns = [
        [0, 2, 6, 8],  # 四个角
        [0, 2, 6],
        [0, 2, 8],
        [0, 6, 8],
        [2, 6, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    
    for x_positions in patterns:
        for _ in range(num_scenarios // len(patterns) + 1):
            board = [0] * 9
            
            # 设置X
            for pos in x_positions:
                board[pos] = 1
            
            # 添加O干扰
            empty = [p for p in range(9) if board[p] == 0]
            random.shuffle(empty)
            num_o = random.randint(1, 3)
            for i in range(min(num_o, len(empty))):
                board[empty[i]] = 2
            
            # 验证实际威胁
            actual_threats = []
            for pos in get_legal_moves(board):
                board[pos] = 1
                if check_win(board, 1):
                    actual_threats.append(pos)
                board[pos] = 0
            
            if len(actual_threats) >= 3:
                fear_label = [0.0] * 9
                for t in actual_threats:
                    fear_label[t] = 1.0
                
                action = random.choice(actual_threats)
                
                scenarios.append((
                    board.copy(),
                    action,
                    -1,
                    fear_label,
                    None,
                    'triple_threat'
                ))
            
            if len(scenarios) >= num_scenarios:
                break
        if len(scenarios) >= num_scenarios:
            break
    
    print(f"   生成 {len(scenarios)} 个三威胁场景")
    return scenarios

# ============ 贪婪场景（可以直接赢）============

def generate_greed_scenarios(num_scenarios=3000):
    """贪婪场景 - 有直接获胜位置"""
    scenarios = []
    
    # X的获胜模式
    x_patterns = [
        ([0,1], 2), ([1,2], 0),
        ([3,4], 5), ([4,5], 3),
        ([6,7], 8), ([7,8], 6),
        ([0,3], 6), ([3,6], 0),
        ([1,4], 7), ([4,7], 1),
        ([2,5], 8), ([5,8], 2),
        ([0,4], 8), ([4,8], 0),
        ([2,4], 6), ([4,6], 2),
    ]
    
    # O的获胜模式
    o_patterns = [
        (0, 4, 8), (2, 4, 6), (6, 4, 2), (8, 4, 0),
        (0, 1, 2), (1, 2, 0), (3, 4, 5), (4, 5, 3),
        (6, 7, 8), (7, 8, 6), (0, 3, 6), (3, 6, 0),
        (1, 4, 7), (4, 7, 1), (2, 5, 8), (5, 8, 2),
    ]
    
    # 生成X的贪婪场景
    for threat_moves, win_move in x_patterns:
        for _ in range(num_scenarios // (len(x_patterns) + len(o_patterns)) + 1):
            board = [0] * 9
            for pos in threat_moves:
                board[pos] = 1
            
            empty = [p for p in range(9) if board[p] == 0 and p != win_move]
            random.shuffle(empty)
            num_o = random.randint(1, 2)
            for i in range(min(num_o, len(empty))):
                board[empty[i]] = 2
            
            board[win_move] = 1
            is_win = check_win(board, 1)
            board[win_move] = 0
            
            if is_win:
                greed_label = [0.0] * 9
                greed_label[win_move] = 1.0
                scenarios.append((
                    board.copy(),
                    win_move,
                    1,
                    None,
                    greed_label,
                    'greed_x'
                ))
    
    # 生成O的贪婪场景
    for pos1, pos2, win_move in o_patterns:
        for _ in range(num_scenarios // (len(x_patterns) + len(o_patterns)) + 1):
            board = [0] * 9
            board[pos1] = 2
            board[pos2] = 2
            
            empty = [p for p in range(9) if board[p] == 0 and p != win_move]
            random.shuffle(empty)
            num_x = random.randint(1, 2)
            for i in range(min(num_x, len(empty))):
                board[empty[i]] = 1
            
            board[win_move] = 2
            is_win = check_win(board, 2)
            board[win_move] = 0
            
            if is_win:
                greed_label = [0.0] * 9
                greed_label[win_move] = 1.0
                scenarios.append((
                    board.copy(),
                    win_move,
                    -1,
                    None,
                    greed_label,
                    'greed_o'
                ))
    
    print(f"   生成 {len(scenarios)} 个贪婪场景")
    return scenarios

# ============ 既恐惧又贪婪的场景 ============

def generate_fear_and_greed_scenarios(num_scenarios=2000):
    """既恐惧又贪婪的场景 - 自己有机会赢，但同时对手也有威胁"""
    scenarios = []
    
    for _ in range(num_scenarios * 2):
        board = [0] * 9
        
        # 随机放3-4个X
        num_x = random.randint(3, 4)
        x_pos = random.sample(range(9), num_x)
        for pos in x_pos:
            board[pos] = 1
        
        # 随机放2-3个O
        empty = [p for p in range(9) if board[p] == 0]
        num_o = random.randint(2, 3)
        if empty:
            o_pos = random.sample(empty, min(num_o, len(empty)))
            for pos in o_pos:
                board[pos] = 2
        
        # 分析局面
        x_wins = analyze_winning_moves(board, 1)
        o_wins = analyze_winning_moves(board, 2)
        x_threats = analyze_board_threats(board, 1)
        o_threats = analyze_board_threats(board, 2)
        
        # 既恐惧又贪婪：自己有获胜机会，同时对手也有威胁
        if (len(x_wins) > 0 and len(o_threats) > 0) or (len(o_wins) > 0 and len(x_threats) > 0):
            fear_label = [0.0] * 9
            greed_label = [0.0] * 9
            
            if len(x_wins) > 0 and len(o_threats) > 0:
                # X的回合
                for t in o_threats:
                    fear_label[t] = 1.0
                for w in x_wins:
                    greed_label[w] = 1.0
                action = random.choice(x_wins)
                value = 1
            else:
                # O的回合
                for t in x_threats:
                    fear_label[t] = 1.0
                for w in o_wins:
                    greed_label[w] = 1.0
                action = random.choice(o_wins)
                value = -1
            
            scenarios.append((
                board.copy(),
                action,
                value,
                fear_label,
                greed_label,
                'fear_and_greed'
            ))
        
        if len(scenarios) >= num_scenarios:
            break
    
    print(f"   生成 {len(scenarios)} 个既恐惧又贪婪场景")
    return scenarios

# ============ 普通场景 ============

def generate_normal_scenarios(num_scenarios=5000):
    """普通场景（从专家数据中提取）"""
    scenarios = []
    
    expert_data = []
    for i in range(num_scenarios // 10):
        start_player = random.choice([1, 2])
        opponent_is_random = random.random() < 0.7
        data = generate_expert_game(
            start_player=start_player, 
            opponent_is_random=opponent_is_random
        )
        expert_data.extend(data)
    
    for board, action, value in expert_data:
        # 确保不是特殊场景
        x_wins = analyze_winning_moves(board, 1)
        o_wins = analyze_winning_moves(board, 2)
        x_threats = analyze_board_threats(board, 1)
        o_threats = analyze_board_threats(board, 2)
        
        is_special = False
        if len(x_wins) > 0 or len(o_wins) > 0 or len(x_threats) > 0 or len(o_threats) > 0:
            is_special = True
        
        if not is_special:
            scenarios.append((board, action, value, None, None, 'normal'))
    
    print(f"   生成 {len(scenarios)} 个普通场景")
    return scenarios

# ============ 主函数 ============

def generate_all_scenarios(
    num_basic=8000,      # 基础威胁（最重要！）
    num_single=3000,
    num_double=2000,
    num_triple=1000,
    num_greed=3000,
    num_fear_greed=2000,
    num_normal=5000
):
    print("=" * 60)
    print("生成完整场景数据集 - 基础威胁优先")
    print("=" * 60)
    
    all_data = []
    
    print(f"\n[1/7] 生成基础威胁场景...")
    basic = generate_basic_threat_scenarios(num_basic)
    all_data.extend(basic)
    
    print(f"\n[2/7] 生成单威胁场景...")
    single = generate_single_threat_scenarios(num_single)
    all_data.extend(single)
    
    print(f"\n[3/7] 生成双威胁场景...")
    double = generate_double_threat_scenarios(num_double)
    all_data.extend(double)
    
    print(f"\n[4/7] 生成三威胁场景...")
    triple = generate_triple_threat_scenarios(num_triple)
    all_data.extend(triple)
    
    print(f"\n[5/7] 生成贪婪场景...")
    greed = generate_greed_scenarios(num_greed)
    all_data.extend(greed)
    
    print(f"\n[6/7] 生成既恐惧又贪婪场景...")
    fear_greed = generate_fear_and_greed_scenarios(num_fear_greed)
    all_data.extend(fear_greed)
    
    print(f"\n[7/7] 生成普通场景...")
    normal = generate_normal_scenarios(num_normal)
    all_data.extend(normal)
    
    random.shuffle(all_data)
    
    # 统计
    print(f"\n📊 最终数据集统计:")
    print(f"   总样本数: {len(all_data)}")
    
    counts = {}
    for item in all_data:
        scene_type = item[5]
        counts[scene_type] = counts.get(scene_type, 0) + 1
    
    for stype, cnt in counts.items():
        print(f"   {stype}: {cnt} ({cnt/len(all_data):.1%})")
    
    output_file = "fear_greed_dataset.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)
    
    print(f"\n✅ 数据集已保存到: {output_file}")
    
    # 打印示例
    print("\n" + "=" * 60)
    print("基础威胁示例:")
    print("=" * 60)
    
    basic_examples = [d for d in all_data if d[5] == 'basic_threat']
    if basic_examples:
        for i in range(min(3, len(basic_examples))):
            board, action, value, fear, _, _ = basic_examples[i]
            print(f"\n示例 {i+1}:")
            print("棋盘:")
            for row in range(3):
                line = ""
                for col in range(3):
                    idx = row * 3 + col
                    if board[idx] == 0:
                        line += f" {idx} "
                    elif board[idx] == 1:
                        line += " X "
                    else:
                        line += " O "
                print(line)
            print(f"威胁位置: {[i for i, v in enumerate(fear) if v > 0]}")
            print(f"正确动作: {action}")

if __name__ == "__main__":
    generate_all_scenarios(
        num_basic=8000,   # 基础威胁 8000
        num_single=3000,
        num_double=2000,
        num_triple=1000,
        num_greed=3000,
        num_fear_greed=2000,
        num_normal=5000
    )
