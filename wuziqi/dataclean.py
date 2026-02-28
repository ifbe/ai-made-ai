# clean_dataset.py
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
        
        # ============ 关键修复：mixed 场景 ============
        if scene_type == 'mixed':
            winning = [i for i, v in enumerate(greed_label) if v > 0] if greed_label is not None else []
            threats = [i for i, v in enumerate(fear_label) if v > 0] if fear_label is not None else []
            
            # 找出既是赢点又是防点的位置
            both = [p for p in winning if p in threats]
            pure_winning = [p for p in winning if p not in both]
            pure_threats = [p for p in threats if p not in both]
            
            # 统计
            stats['mixed_both'] += len(both)
            stats['mixed_pure_winning'] += len(pure_winning)
            stats['mixed_pure_threats'] += len(pure_threats)
            
            # 1. 如果有 both 位置，必须选 both！
            if both and action not in both:
                if fix:
                    new_action = random.choice(both)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复 mixed #{idx}: 存在必争之地，动作从 {pos_to_str(action)} 改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除 mixed #{idx}: 存在必争之地却选其他")
                    continue
                else:
                    stats['mixed_wrong_action'] += 1
                    print(f"  📊 分析: mixed #{idx} 存在必之地却选 {pos_to_str(action)}")
            
            # 2. 如果没有 both，但有纯赢点，必须选纯赢点
            elif pure_winning and action in pure_threats:
                if fix:
                    new_action = random.choice(pure_winning)
                    fixed_item[2] = new_action
                    need_fix = True
                    fixed_count += 1
                    print(f"  ✅ 修复 mixed #{idx}: 有赢点却选防守，改为 {pos_to_str(new_action)}")
                elif delete:
                    deleted_count += 1
                    print(f"  🗑️ 删除 mixed #{idx}: 有赢点却选防守")
                    continue
                else:
                    stats['mixed_wrong_action'] += 1
                    print(f"  📊 分析: mixed #{idx} 有赢点却选防守点 {pos_to_str(action)}")
            
            # 3. 只有防点的情况（正常）
            elif not pure_winning and not both and action in pure_threats:
                # 这是正确的，什么也不做
                pass
        
        # ============ 检查恐惧标签 ============
        if scene_type in ['fear', 'mixed'] and fear_label is not None:
            fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
            valid_fears = [p for p in fear_positions if is_real_threat(board, player, p)]
            invalid_fears = [p for p in fear_positions if p not in valid_fears]
            
            stats['valid_fear'] += len(valid_fears)
            stats['invalid_fear'] += len(invalid_fears)
            
            if invalid_fears:
                if fix:
                    # 修复无效的恐惧标签
                    new_fear = [0.0] * BOARD_POSITIONS
                    for pos in valid_fears:
                        new_fear[pos] = 1.0
                    fixed_item[4] = new_fear
                    need_fix = True
                    print(f"  🔧 修复恐惧标签 {idx}: 移除 {len(invalid_fears)} 个无效标记")
                elif delete:
                    # 删除模式，但这里不直接删除，后面会判断
                    pass
                else:
                    stats['invalid_fear_samples'] += 1
                    print(f"  📊 分析: #{idx} 有 {len(invalid_fears)} 个无效恐惧标记")
        
        # ============ 检查贪婪标签 ============
        if scene_type in ['greed', 'mixed'] and greed_label is not None:
            greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
            valid_greed = [p for p in greed_positions if is_real_winning_move(board, player, p)]
            invalid_greed = [p for p in greed_positions if p not in valid_greed]
            
            stats['valid_greed'] += len(valid_greed)
            stats['invalid_greed'] += len(invalid_greed)
            
            if invalid_greed:
                if fix:
                    # 修复无效的贪婪标签
                    new_greed = [0.0] * BOARD_POSITIONS
                    for pos in valid_greed:
                        new_greed[pos] = 1.0
                    fixed_item[5] = new_greed
                    need_fix = True
                    print(f"  🔧 修复贪婪标签 {idx}: 移除 {len(invalid_greed)} 个无效标记")
                elif delete:
                    # 删除模式，但这里不直接删除，后面会判断
                    pass
                else:
                    stats['invalid_greed_samples'] += 1
                    print(f"  📊 分析: #{idx} 有 {len(invalid_greed)} 个无效贪婪标记")
        
        # 如果场景类型是 fear/greed 但修复后没有有效标记，则删除
        if scene_type == 'fear' and fixed_item[4] is not None and sum(fixed_item[4]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除 fear 样本 #{idx}: 无有效恐惧标记")
                continue
            elif fix:
                print(f"  ⚠️ fear #{idx} 无有效恐惧标记，无法修复")
        
        if scene_type == 'greed' and fixed_item[5] is not None and sum(fixed_item[5]) == 0:
            if delete:
                deleted_count += 1
                print(f"  🗑️ 删除 greed 样本 #{idx}: 无有效贪婪标记")
                continue
            elif fix:
                print(f"  ⚠️ greed #{idx} 无有效贪婪标记，无法修复")
        
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
    
    for stype in ['fear', 'greed', 'mixed', 'normal']:
        cnt = final_counts.get(stype, 0)
        print(f"  {stype}: {cnt}")
    
    print(f"\n混合场景分析:")
    print(f"  必争之地(both): {stats['mixed_both']}")
    print(f"  纯赢点: {stats['mixed_pure_winning']}")
    print(f"  纯防点: {stats['mixed_pure_threats']}")
    print(f"  错误动作: {stats['mixed_wrong_action']}")
    
    print(f"\n标记质量:")
    print(f"  有效恐惧标记: {stats['valid_fear']}")
    print(f"  无效恐惧标记: {stats['invalid_fear']}")
    print(f"  有效贪婪标记: {stats['valid_greed']}")
    print(f"  无效贪婪标记: {stats['invalid_greed']}")
    
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
    parser = argparse.ArgumentParser(description='清理五子棋数据集')
    parser.add_argument('input', default='wuziqi_dataset_real.pkl', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--fix', action='store_true', help='修复问题样本')
    parser.add_argument('--del', dest='delete', action='store_true', help='删除问题样本')
    
    args = parser.parse_args()
    
    analyze_dataset(args.input, args.output, fix=args.fix, delete=args.delete)
