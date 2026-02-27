# clean_dataset.py
import pickle
import argparse
import shutil
from collections import defaultdict
from game import (
    check_win, get_legal_moves, 
    BOARD_SIZE, BLACK, WHITE, EMPTY, BOARD_POSITIONS,
    pos_to_str
)

def is_real_winning_move(board, player, pos):
    if board[pos] != EMPTY:
        return False
    board[pos] = player
    result = check_win(board, player)
    board[pos] = EMPTY
    return result

def is_real_threat(board, player, pos):
    opponent = 3 - player
    if board[pos] != EMPTY:
        return False
    board[pos] = opponent
    result = check_win(board, opponent)
    board[pos] = EMPTY
    return result

def analyze_dataset(input_file, output_file=None, delete=False):
    print(f"加载数据集: {input_file}")
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"总样本数: {len(data)} 条")
    print("=" * 60)
    
    stats = defaultdict(int)
    issues = []
    
    for idx, item in enumerate(data):
        if len(item) < 7:
            stats['invalid_format'] += 1
            issues.append(f"样本 #{idx}: 格式错误")
            continue
            
        board, player, action, value, fear_label, greed_label, scene_type = item[:7]
        
        if player not in [BLACK, WHITE]:
            issues.append(f"样本 #{idx}: 无效玩家值 {player}")
            stats['invalid_player'] += 1
            continue
        
        legals = get_legal_moves(board)
        if action not in legals:
            issues.append(f"样本 #{idx} ({scene_type}): 动作 {pos_to_str(action)} 不合法")
            stats['illegal_action'] += 1
            continue
        
        if scene_type in ['fear', 'mixed'] and fear_label is not None:
            fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
            valid_fears = []
            invalid_fears = []
            
            for pos in fear_positions:
                if is_real_threat(board, player, pos):
                    valid_fears.append(pos)
                else:
                    invalid_fears.append(pos)
            
            if invalid_fears:
                issues.append(f"样本 #{idx} ({scene_type}): 恐惧标记 {[pos_to_str(p) for p in invalid_fears]} 无效")
                stats['invalid_fear'] += len(invalid_fears)
            
            if valid_fears:
                stats['valid_fear'] += len(valid_fears)
        
        if scene_type in ['greed', 'mixed'] and greed_label is not None:
            greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
            valid_greed = []
            invalid_greed = []
            
            for pos in greed_positions:
                if is_real_winning_move(board, player, pos):
                    valid_greed.append(pos)
                else:
                    invalid_greed.append(pos)
            
            if invalid_greed:
                issues.append(f"样本 #{idx} ({scene_type}): 贪婪标记 {[pos_to_str(p) for p in invalid_greed]} 无效")
                stats['invalid_greed'] += len(invalid_greed)
            
            if valid_greed:
                stats['valid_greed'] += len(valid_greed)
        
        stats[scene_type] += 1
        stats[f'player_{player}'] += 1
    
    print("\n📊 数据集统计:")
    print(f"   恐惧场景: {stats['fear']}")
    print(f"   贪婪场景: {stats['greed']}")
    print(f"   混合场景: {stats['mixed']}")
    print(f"   普通场景: {stats['normal']}")
    
    print(f"\n👥 玩家分布:")
    print(f"   黑棋回合: {stats['player_1']}")
    print(f"   白棋回合: {stats['player_2']}")
    
    print(f"\n🏷️ 标记质量:")
    print(f"   有效恐惧标记: {stats['valid_fear']}")
    print(f"   无效恐惧标记: {stats['invalid_fear']}")
    print(f"   有效贪婪标记: {stats['valid_greed']}")
    print(f"   无效贪婪标记: {stats['invalid_greed']}")
    
    if stats['illegal_action'] > 0:
        print(f"\n❌ 非法动作: {stats['illegal_action']} 个")
    
    if stats['invalid_player'] > 0:
        print(f"❌ 无效玩家: {stats['invalid_player']} 个")
    
    if issues:
        print(f"\n⚠️ 发现 {len(issues)} 个问题:")
        for issue in issues[:20]:
            print(f"   {issue}")
        if len(issues) > 20:
            print(f"   ... 还有 {len(issues)-20} 个问题")
    else:
        print("\n✅ 没有发现问题，数据质量良好！")
    
    if delete and issues:
        print(f"\n🗑️ 删除模式已开启...")
        
        cleaned_data = []
        deleted_count = 0
        fixed_count = 0
        
        for idx, item in enumerate(data):
            if len(item) < 7:
                deleted_count += 1
                continue
                
            board, player, action, value, fear_label, greed_label, scene_type = item[:7]
            
            if player not in [BLACK, WHITE]:
                deleted_count += 1
                continue
            
            legals = get_legal_moves(board)
            if action not in legals:
                deleted_count += 1
                continue
            
            fixed_item = list(item)
            need_fix = False
            
            if scene_type in ['fear', 'mixed'] and fear_label is not None:
                fear_positions = [i for i, v in enumerate(fear_label) if v > 0]
                valid_fears = [p for p in fear_positions if is_real_threat(board, player, p)]
                
                if len(valid_fears) < len(fear_positions):
                    new_fear = [0.0] * BOARD_POSITIONS
                    for pos in valid_fears:
                        new_fear[pos] = 1.0
                    fixed_item[4] = new_fear
                    need_fix = True
            
            if scene_type in ['greed', 'mixed'] and greed_label is not None:
                greed_positions = [i for i, v in enumerate(greed_label) if v > 0]
                valid_greed = [p for p in greed_positions if is_real_winning_move(board, player, p)]
                
                if len(valid_greed) < len(greed_positions):
                    new_greed = [0.0] * BOARD_POSITIONS
                    for pos in valid_greed:
                        new_greed[pos] = 1.0
                    fixed_item[5] = new_greed
                    need_fix = True
            
            if scene_type == 'fear' and fixed_item[4] is not None and sum(fixed_item[4]) == 0:
                deleted_count += 1
                continue
            if scene_type == 'greed' and fixed_item[5] is not None and sum(fixed_item[5]) == 0:
                deleted_count += 1
                continue
            
            if need_fix:
                cleaned_data.append(tuple(fixed_item))
                fixed_count += 1
            else:
                cleaned_data.append(item)
        
        print(f"   删除了 {deleted_count} 个无效样本")
        print(f"   修复了 {fixed_count} 个样本的标记")
        print(f"   剩余 {len(cleaned_data)} 个样本")
        
        output_path = output_file if output_file else input_file
        if output_path == input_file:
            backup = input_file + '.bak'
            shutil.copy2(input_file, backup)
            print(f"   原文件已备份到: {backup}")
        
        with open(output_path, "wb") as f:
            pickle.dump(cleaned_data, f)
        
        print(f"✅ 已保存到: {output_path}")
    
    return stats, issues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='wuziqi_dataset_real.pkl', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    
    analyze_dataset(args.input, args.output, delete=args.delete)
