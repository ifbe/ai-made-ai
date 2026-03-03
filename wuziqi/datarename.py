# rename_scenes.py
import pickle
import argparse
from collections import defaultdict

def rename_scenes(input_file, output_file):
    """将场景类型从 greed/fear 重命名为 winning/losing"""
    print(f"加载数据集: {input_file}")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据: {len(data)} 条")
    
    # 统计改名前的分布
    old_counts = defaultdict(int)
    renamed_data = []
    
    for item in data:
        if len(item) >= 7:
            board, player, action, value, fear_label, greed_label, scene_type = item[:7]
            
            # 记录原类型
            old_counts[scene_type] += 1
            
            # 重命名
            if scene_type == 'greed':
                new_scene_type = 'winning'
            elif scene_type == 'fear':
                new_scene_type = 'losing'
            else:
                new_scene_type = scene_type  # mixed/normal 保持不变
            
            # 重新打包
            new_item = (
                board, player, action, value, 
                fear_label, greed_label, new_scene_type
            )
            renamed_data.append(new_item)
        else:
            renamed_data.append(item)
    
    # 统计改名后的分布
    new_counts = defaultdict(int)
    for item in renamed_data:
        if len(item) >= 7:
            new_counts[item[6]] += 1
    
    print("\n📊 改名前后对比:")
    print(f"{'类型':<10} {'改名前':<10} {'改名后':<10}")
    print("-" * 35)
    for stype in ['greed', 'fear', 'winning', 'losing', 'mixed', 'normal']:
        old = old_counts.get(stype, 0)
        new = new_counts.get(stype, 0)
        if old > 0 or new > 0:
            print(f"{stype:<10} {old:<10} {new:<10}")
    
    # 保存
    with open(output_file, 'wb') as f:
        pickle.dump(renamed_data, f)
    
    print(f"\n✅ 已保存到: {output_file}")
    print(f"   总样本数: {len(renamed_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='重命名场景类型: greed→winning, fear→losing')
    parser.add_argument('input', default='wuziqi_dataset_real.pkl', nargs='?')
    parser.add_argument('output', default='wuziqi_dataset_renamed.pkl', nargs='?')
    args = parser.parse_args()
    
    rename_scenes(args.input, args.output)
