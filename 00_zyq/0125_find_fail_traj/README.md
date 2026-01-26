# Episode数据条目查看结果

本目录包含从数据集中提取的特定episode的详细信息。

## 文件说明

### `episode_70_data.json`
完整的数据条目JSON文件，包含所有原始数据字段：
- episode_id, trajectory_id, scene_id
- start_position, start_rotation
- goals (目标位置和半径)
- reference_path (参考路径的所有点)
- instruction (指令文本和token)
- info (包含geodesic_distance等元信息)

### `episode_summary.md`
数据条目的详细摘要，包含：
- 基本信息
- 起始位置和方向
- 目标位置
- 距离信息
- 参考路径
- 指令内容

### `inspect_dataset_episode.py`
用于查看数据集episode的Python脚本。可以：
- 从数据集中查找指定的scene_id和episode_id
- 显示episode的详细信息
- 保存找到的episode数据到JSON文件

## 本次查看的Episode信息

- **scene_id**: `2azQ1b91cZZ`
- **episode_id**: `71`
- **trajectory_id**: `254`
- **数据集**: `val_unseen`
- **数据集索引**: `70`

## 使用方法

如果需要查看其他episode，可以修改脚本中的参数：

```python
scene_id = "2azQ1b91cZZ"
episode_id = 71
```

然后运行：
```bash
python3 inspect_dataset_episode.py
```


