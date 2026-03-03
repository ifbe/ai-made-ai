# clean.py
import pickle
import argparse
import random
import numpy as np
import shutil
from collections import defaultdict
from game import (
    check_win, get_legal_moves, 
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS,
    pos_to_str
)

def is_real_winning_move(board, player, pos):
    """检查是否是真正的获胜机会 - 下这里能直接赢"""
    if board[pos] != EMPTY:
        return False
    board[pos] = player
    result = check_win(board, player)
    board[pos] = EMPTY
    return result

def is_real_threat(board, player, pos):
    """检查是否是真正的威胁 - 对手下这里能直接赢"""
    opponent = 3 - player
    if board[pos] != EMPTY:
        return False
    board[pos] = opponent
    result = check_win(board, opponent)
    board[pos] = EMPTY
    return result

def has_four_in_row(board, player, pos):
    """检查下在pos后是否形成4连（允许更多，只要不是5连即可）"""
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
        
        # 只要 >=4 就算，但还要检查是否5连
        if count >= 4:
            # 检查是否形成5连（不应该发生）
            # 简单起见，只要不是明显的5连就通过
            return True
    
    return False

def analyze_dataset(input_file, output_file=None, fix=False, delete=False):
    """
    分析并修复数据集
    fix=True: 修复问题样本
    delete=True: 删除问题样本
    fix=False and delete=False: 只分析不修改
    """
    print(f"加载数据集: {input_file}")
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"原始数据: {len(data)} 条")
    print("=" * 60)
    
    stats = defaultdict(int)
    cleaned_data = []
    fixed_count = 0
    deleted_count = 0
    
    for idx, item in enumerate(data):
        if len(item) < 7:
            stats['invalid_format'] += 1
            if delete:
                deleted_count += 1
            else:
                cleaned_data.append(item)
            continue
            
        board, player, action, value, fear_label, greed_label, scene_type = item[:7]
        
        # 确保player是整数
        if isinstance(player, (list, tuple, np.ndarray)):
            player = player[0] if len(player) > 0 else BLACK
        player = int(player) if player is not None else BLACK
        
        # 检查动作合法性
        legals = get_legal_moves(board)
        if action not in legals:
            stats['illegal_action'] += 1
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除样本 #{idx}: 动作 {pos_to_str(action)} 不合法")
            elif fix:
                print(f"  ⚠️ 样本 #{idx} 动作不合法，无法修复")
                cleaned_data.append(item)
            else:
                cleaned_data.append(item)
            continue
        
        # 创建可修改的副本
        fixed_item = list(item)
        need_fix = False
        
        # ============ 检查winning/losing场景（直接赢/直接输）============
        if scene_type in ['winning', 'losing']:
            if scene_type == 'winning' and greed_label is not None:
                win_positions = [i for i, v in enumerate(greed_label) if v > 0]
                valid_wins = [p for p in win_positions if is_real_winning_move(board, player, p)]
                invalid_wins = [p for p in win_positions if p not in valid_wins]
                
                stats['valid_win'] += len(valid_wins)
                stats['invalid_win'] += len(invalid_wins)
                
                if invalid_wins and fix:
                    new_greed = [0.0] * BOARD_POSITIONS
                    for pos in valid_wins:
                        new_greed[pos] = 1.0
                    fixed_item[5] = new_greed
                    need_fix = True
                    print(f"  🔧 修复winning样本 #{idx}: 移除 {len(invalid_wins)} 个无效赢点")
            
            elif scene_type == 'losing' and fear_label is not None:
                threat_positions = [i for i, v in enumerate(fear_label) if v > 0]
                valid_threats = [p for p in threat_positions if is_real_threat(board, player, p)]
                invalid_threats = [p for p in threat_positions if p not in valid_threats]
                
                stats['valid_lose'] += len(valid_threats)
                stats['invalid_lose'] += len(invalid_threats)
                
                if invalid_threats and fix:
                    new_fear = [0.0] * BOARD_POSITIONS
                    for pos in valid_threats:
                        new_fear[pos] = 1.0
                    fixed_item[4] = new_fear
                    need_fix = True
                    print(f"  🔧 修复losing样本 #{idx}: 移除 {len(invalid_threats)} 个无效威胁")
        
        # ============ 检查greed/fear场景（4连场景）============
        elif scene_type == 'greed' and greed_label is not None:
            greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
            valid_greed = []
            invalid_greed = []
            
            for pos in greed_positions:
                board[pos] = player
                if has_four_in_row(board, player, pos) and not check_win(board, player):
                    valid_greed.append(pos)
                else:
                    invalid_greed.append(pos)
                board[pos] = EMPTY
            
            stats['valid_greed'] += len(valid_greed)
            stats['invalid_greed'] += len(invalid_greed)
            
            if invalid_greed and fix:
                new_greed = [0.0] * BOARD_POSITIONS
                for pos in valid_greed:
                    new_greed[pos] = 1.0
                fixed_item[5] = new_greed
                need_fix = True
                print(f"  🔧 修复greed样本 #{idx}: 移除 {len(invalid_greed)} 个无效4连标记")
            
            # 检查动作是否在有效4连中
            if action not in valid_greed and valid_greed:
                if fix:
                    new_action = random.choice(valid_greed)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复greed动作 #{idx}: 从 {pos_to_str(action)} 改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除greed样本 #{idx}: 动作不在4连中")
                    continue
        
        elif scene_type == 'fear' and fear_label is not None:
            fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
            valid_fear = []
            invalid_fear = []
            opponent = 3 - player
            
            for pos in fear_positions:
                board[pos] = opponent
                if has_four_in_row(board, opponent, pos) and not check_win(board, opponent):
                    valid_fear.append(pos)
                else:
                    invalid_fear.append(pos)
                board[pos] = EMPTY
            
            stats['valid_fear'] += len(valid_fear)
            stats['invalid_fear'] += len(invalid_fear)
            
            if invalid_fear and fix:
                new_fear = [0.0] * BOARD_POSITIONS
                for pos in valid_fear:
                    new_fear[pos] = 1.0
                fixed_item[4] = new_fear
                need_fix = True
                print(f"  🔧 修复fear样本 #{idx}: 移除 {len(invalid_fear)} 个无效4连标记")
            
            # 检查动作是否在有效4连中
            if action not in valid_fear and valid_fear:
                if fix:
                    new_action = random.choice(valid_fear)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复fear动作 #{idx}: 从 {pos_to_str(action)} 改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除fear样本 #{idx}: 动作不在4连中")
                    continue
        
        # ============ 检查mixed场景 ============
        elif scene_type == 'mixed':
            winning = [i for i, v in enumerate(greed_label) if v > 0] if greed_label is not None else []
            threats = [i for i, v in enumerate(fear_label) if v > 0] if fear_label is not None else []
            
            # 找出既是赢点又是防点的位置
            both = [p for p in winning if p in threats]
            pure_winning = [p for p in winning if p not in both]
            pure_threats = [p for p in threats if p not in both]
            
            stats['mixed_both'] += len(both)
            stats['mixed_pure_winning'] += len(pure_winning)
            stats['mixed_pure_threats'] += len(pure_threats)
            
            # 如果有both位置，必须选both
            if both and action not in both:
                if fix:
                    new_action = random.choice(both)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复mixed #{idx}: 存在必争之地，改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除mixed #{idx}: 存在必争之地却选其他")
                    continue
            # 如果没有both，但有纯赢点，必须选纯赢点
            elif pure_winning and action in pure_threats:
                if fix:
                    new_action = random.choice(pure_winning)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复mixed #{idx}: 有赢点却选防守，改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除mixed #{idx}: 有赢点却选防守")
                    continue
        
        # 如果场景类型是winning/losing但修复后没有有效标记，则删除
        if scene_type == 'winning' and fixed_item[5] is not None and sum(fixed_item[5]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除winning样本 #{idx}: 无有效赢点")
                continue
        
        if scene_type == 'losing' and fixed_item[4] is not None and sum(fixed_item[4]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除losing样本 #{idx}: 无有效威胁")
                continue
        
        if scene_type == 'greed' and fixed_item[5] is not None and sum(fixed_item[5]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除greed样本 #{idx}: 无有效4连")
                continue
        
        if scene_type == 'fear' and fixed_item[4] is not None and sum(fixed_item[4]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除fear样本 #{idx}: 无有效4连")
                continue
        
        # 保存处理后的样本
        if need_fix:
            cleaned_data.append(tuple(fixed_item))
        else:
            cleaned_data.append(item)
        
        stats[scene_type] += 1
    
    # 打印统计
    print(f"\n📊 处理结果:")
    print(f"   保留: {len(cleaned_data)} 条")
    if fix:
        print(f"   修复: {fixed_count} 条")
    if delete:
        print(f"   删除: {deleted_count} 条")
    
    print(f"\n场景分布:")
    final_counts = defaultdict(int)
    for item in cleaned_data:
        if len(item) >= 7:
            final_counts[item[6]] += 1
    
    for stype in ['winning', 'losing', 'greed', 'fear', 'mixed', 'normal']:
        cnt = final_counts.get(stype, 0)
        print(f"  {stype}: {cnt}")
    
    print(f"\n标记质量:")
    print(f"  winning有效赢点: {stats['valid_win']}")
    print(f"  winning无效赢点: {stats['invalid_win']}")
    print(f"  losing有效威胁: {stats['valid_lose']}")
    print(f"  losing无效威胁: {stats['invalid_lose']}")
    print(f"  greed有效4连: {stats['valid_greed']}")
    print(f"  greed无效4连: {stats['invalid_greed']}")
    print(f"  fear有效4连: {stats['valid_fear']}")
    print(f"  fear无效4连: {stats['invalid_fear']}")
    
    # 保存
    if fix or delete:
        output_path = output_file if output_file else input_file
        if output_path == input_file:
            backup = input_file + '.bak'
            shutil.copy2(input_file, backup)
            print(f"\n📦 原文件已备份: {backup}")
        
        with open(output_path, "wb") as f:
            pickle.dump(cleaned_data, f)
        
        print(f"✅ 已保存到: {output_path}")
    else:
        print(f"\n🔍 分析模式: 未修改文件")
    
    return cleaned_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='清理五子棋数据集 (6场景版)')
    parser.add_argument('input', default='wuziqi_dataset_real.pkl', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--fix', action='store_true', help='修复问题样本')
    parser.add_argument('--del', dest='delete', action='store_true', help='删除问题样本')
    
    args = parser.parse_args()
    
    analyze_dataset(args.input, args.output, fix=args.fix, delete=args.delete)
