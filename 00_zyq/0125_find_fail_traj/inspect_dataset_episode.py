#!/usr/bin/env python3
"""
查看数据集中指定episode的具体数据条目
"""
import json
import gzip
import os
from pathlib import Path

def load_and_find_episode(dataset_path, scene_id, episode_id):
    """
    从数据集中查找指定的episode数据
    
    Args:
        dataset_path: 数据集文件路径
        scene_id: 场景ID
        episode_id: episode ID
    """
    print(f"正在读取数据集: {dataset_path}")
    print(f"查找 scene_id={scene_id}, episode_id={episode_id}")
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return None
    
    # 读取数据集
    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    print(f"数据集中共有 {len(episodes)} 个episodes")
    
    # 查找匹配的episode
    found_episodes = []
    for idx, episode in enumerate(episodes):
        ep_scene_id = episode.get('scene_id', '')
        ep_episode_id = episode.get('episode_id', -1)
        
        # 处理scene_id格式: 可能是完整路径如 'mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb' 或只是 '2azQ1b91cZZ'
        scene_match = False
        if '/' in ep_scene_id:
            # 提取scene_id部分 (例如从 'mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb' 提取 '2azQ1b91cZZ')
            ep_scene_base = ep_scene_id.split('/')[1] if len(ep_scene_id.split('/')) > 1 else ep_scene_id
            scene_match = ep_scene_base == scene_id
        else:
            scene_match = ep_scene_id == scene_id
        
        if scene_match and ep_episode_id == episode_id:
            found_episodes.append((idx, episode))
            print(f"\n找到匹配的episode (索引 {idx}):")
            print(f"  scene_id: {ep_scene_id}")
            print(f"  episode_id: {ep_episode_id}")
    
    if not found_episodes:
        print(f"\n未找到匹配的episode!")
        print(f"\n数据集中的scene_id示例 (前10个):")
        for i, ep in enumerate(episodes[:10]):
            print(f"  [{i}] scene_id={ep.get('scene_id', 'N/A')}, episode_id={ep.get('episode_id', 'N/A')}")
        
        # 统计scene_id分布
        scene_ids = {}
        for ep in episodes:
            sid = ep.get('scene_id', '')
            if '/' in sid:
                sid_base = sid.split('/')[1] if len(sid.split('/')) > 1 else sid
            else:
                sid_base = sid
            scene_ids[sid_base] = scene_ids.get(sid_base, 0) + 1
        
        print(f"\n数据集中的scene_id统计 (共{len(scene_ids)}个不同的scene):")
        if scene_id in scene_ids:
            print(f"  ✓ 找到目标scene_id '{scene_id}'，共有 {scene_ids[scene_id]} 个episodes")
            # 列出该scene的所有episode_id
            ep_ids = [ep.get('episode_id') for ep in episodes 
                     if (ep.get('scene_id', '').split('/')[1] if '/' in ep.get('scene_id', '') else ep.get('scene_id', '')) == scene_id]
            print(f"  该scene的episode_id范围: {min(ep_ids)} - {max(ep_ids)}")
        else:
            print(f"  ✗ 未找到目标scene_id '{scene_id}'")
            print(f"  可用的scene_id示例: {list(scene_ids.keys())[:10]}")
    
    return found_episodes

def save_episode_data(episodes, output_dir):
    """
    保存episode数据到文件
    
    Args:
        episodes: 找到的episode列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (orig_idx, episode) in enumerate(episodes):
        # 保存为JSON格式
        output_file = os.path.join(output_dir, f"episode_{orig_idx}_data.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(episode, f, indent=2, ensure_ascii=False)
        
        print(f"\n已保存episode数据到: {output_file}")
        
        # 打印关键信息
        print(f"\nEpisode关键信息:")
        print(f"  trajectory_id: {episode.get('trajectory_id', 'N/A')}")
        print(f"  episode_id: {episode.get('episode_id', 'N/A')}")
        print(f"  scene_id: {episode.get('scene_id', 'N/A')}")
        print(f"  start_position: {episode.get('start_position', 'N/A')}")
        print(f"  start_rotation: {episode.get('start_rotation', 'N/A')}")
        if 'reference_path' in episode:
            print(f"  reference_path长度: {len(episode['reference_path'])}")
            print(f"  reference_path前3个点: {episode['reference_path'][:3]}")
        if 'instructions' in episode:
            print(f"  instructions: {episode['instructions']}")
        if 'instruction' in episode:
            print(f"  instruction: {episode['instruction']}")

def main():
    # 从配置文件读取参数
    # 根据 vln_r2r.yaml，数据集路径是 data/vln_ce/raw_data/r2r/{split}/{split}.json.gz
    # split是 val_unseen
    workspace_root = Path(__file__).parent.parent.parent
    dataset_root = workspace_root / "data" / "vln_ce" / "raw_data" / "r2r"
    split = "val_unseen"
    dataset_path = dataset_root / split / f"{split}.json.gz"
    
    # 从配置文件读取目标episode
    scene_id = "2azQ1b91cZZ"
    episode_id = 71
    
    print("=" * 80)
    print("数据集Episode查看工具")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print(f"目标: scene_id={scene_id}, episode_id={episode_id}")
    print("=" * 80)
    
    # 查找episode
    found_episodes = load_and_find_episode(str(dataset_path), scene_id, episode_id)
    
    if found_episodes:
        # 保存到输出目录
        output_dir = Path(__file__).parent
        save_episode_data(found_episodes, str(output_dir))
        print(f"\n✓ 成功找到并保存了 {len(found_episodes)} 个匹配的episode")
    else:
        print(f"\n✗ 未找到匹配的episode，请检查scene_id和episode_id是否正确")

if __name__ == "__main__":
    main()


