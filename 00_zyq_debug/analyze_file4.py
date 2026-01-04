#!/usr/bin/env python
"""
详细分析 model-00004-of-00004.safetensors 文件中的权重
"""
import sys
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))


def analyze_weight(weight, key):
    """分析单个权重的统计信息"""
    non_zero = (weight != 0).sum().item()
    total = weight.numel()
    mean_val = weight.mean().item()
    std_val = weight.std().item()
    max_val = weight.max().item()
    min_val = weight.min().item()
    sum_val = weight.sum().item()
    
    return {
        'key': key,
        'shape': weight.shape,
        'dtype': str(weight.dtype),
        'total': total,
        'non_zero': non_zero,
        'zero_count': total - non_zero,
        'zero_ratio': (total - non_zero) / total if total > 0 else 0,
        'mean': mean_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'sum': sum_val,
        'is_all_zero': non_zero == 0,
    }


def main():
    """主函数"""
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    weight_file = model_path / "model-00004-of-00004.safetensors"
    
    print("=" * 80)
    print("详细分析 model-00004-of-00004.safetensors")
    print("=" * 80)
    print(f"文件路径: {weight_file}")
    print(f"文件存在: {weight_file.exists()}")
    
    if not weight_file.exists():
        print("✗ 文件不存在!")
        return
    
    # 获取文件大小
    file_size = weight_file.stat().st_size
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    print()
    
    # 打开文件
    print("正在读取权重文件...")
    all_weights = []
    zero_weights = []
    non_zero_weights = []
    
    with safe_open(str(weight_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"总权重数量: {len(keys)}")
        print()
        
        # 分析每个权重
        print("=" * 80)
        print("分析所有权重...")
        print("=" * 80)
        
        for i, key in enumerate(keys):
            weight = f.get_tensor(key)
            stats = analyze_weight(weight, key)
            all_weights.append(stats)
            
            if stats['is_all_zero']:
                zero_weights.append(stats)
            else:
                non_zero_weights.append(stats)
            
            # 每100个显示进度
            if (i + 1) % 100 == 0:
                print(f"  已分析 {i + 1}/{len(keys)} 个权重...")
        
        print(f"✓ 分析完成!")
        print()
    
    # 统计信息
    print("=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"总权重数量: {len(all_weights)}")
    print(f"全为 0 的权重: {len(zero_weights)} ({len(zero_weights)/len(all_weights)*100:.1f}%)")
    print(f"包含非零值的权重: {len(non_zero_weights)} ({len(non_zero_weights)/len(all_weights)*100:.1f}%)")
    print()
    
    # 按模块分类统计
    print("=" * 80)
    print("按模块分类统计")
    print("=" * 80)
    
    module_stats = defaultdict(lambda: {'total': 0, 'zero': 0, 'non_zero': 0})
    
    for stats in all_weights:
        # 提取模块名（第一个点之前的部分）
        parts = stats['key'].split('.')
        if len(parts) >= 2:
            module = '.'.join(parts[:2])  # 例如 "model.layers" 或 "model.navdp"
        else:
            module = parts[0]
        
        module_stats[module]['total'] += 1
        if stats['is_all_zero']:
            module_stats[module]['zero'] += 1
        else:
            module_stats[module]['non_zero'] += 1
    
    # 按模块显示
    for module in sorted(module_stats.keys()):
        stats = module_stats[module]
        zero_ratio = stats['zero'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "⚠️" if zero_ratio > 50 else "✓"
        print(f"{status} {module}:")
        print(f"    总数: {stats['total']}, 全0: {stats['zero']} ({zero_ratio:.1f}%), 正常: {stats['non_zero']}")
    print()
    
    # 显示全为0的权重（按模块分组）
    print("=" * 80)
    print("全为 0 的权重详情（按模块分组）")
    print("=" * 80)
    
    zero_by_module = defaultdict(list)
    for stats in zero_weights:
        parts = stats['key'].split('.')
        if len(parts) >= 2:
            module = '.'.join(parts[:2])
        else:
            module = parts[0]
        zero_by_module[module].append(stats)
    
    for module in sorted(zero_by_module.keys()):
        weights = zero_by_module[module]
        print(f"\n{module} ({len(weights)} 个全0权重):")
        for stats in weights[:20]:  # 每个模块最多显示20个
            print(f"  ✗ {stats['key']}: shape={stats['shape']}")
        if len(weights) > 20:
            print(f"  ... 还有 {len(weights) - 20} 个全0权重")
    print()
    
    # 显示包含非零值的权重
    print("=" * 80)
    print("包含非零值的权重详情")
    print("=" * 80)
    
    if non_zero_weights:
        for stats in non_zero_weights:
            print(f"✓ {stats['key']}:")
            print(f"    Shape: {stats['shape']}")
            print(f"    Dtype: {stats['dtype']}")
            print(f"    Non-zero: {stats['non_zero']} / {stats['total']} ({stats['non_zero']/stats['total']*100:.2f}%)")
            print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"    Max: {stats['max']:.6f}, Min: {stats['min']:.6f}")
            print(f"    Sum: {stats['sum']:.6f}")
            print()
    else:
        print("✗ 没有找到包含非零值的权重!")
        print()
    
    # 特别关注的关键层
    print("=" * 80)
    print("关键层检查")
    print("=" * 80)
    
    critical_keys = [
        "model.norm.weight",
        "model.layers.27.input_layernorm.weight",
        "model.layers.27.post_attention_layernorm.weight",
        "model.layers.27.self_attn.q_proj.weight",
        "model.layers.27.self_attn.k_proj.weight",
        "model.layers.27.self_attn.v_proj.weight",
        "model.layers.27.self_attn.o_proj.weight",
        "model.layers.27.mlp.gate_proj.weight",
        "model.layers.27.mlp.up_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
        "lm_head.weight",
    ]
    
    for key in critical_keys:
        found = False
        for stats in all_weights:
            if stats['key'] == key:
                found = True
                status = "✗" if stats['is_all_zero'] else "✓"
                print(f"{status} {key}:")
                print(f"    Shape: {stats['shape']}")
                print(f"    Non-zero: {stats['non_zero']} / {stats['total']}")
                print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                if stats['is_all_zero']:
                    print(f"    ⚠️  全为 0 - 这是问题所在!")
                print()
                break
        if not found:
            print(f"✗ {key}: 未找到")
            print()
    
    # 总结
    print("=" * 80)
    print("问题总结")
    print("=" * 80)
    print(f"文件: model-00004-of-00004.safetensors")
    print(f"总权重数: {len(all_weights)}")
    print(f"全为0的权重数: {len(zero_weights)} ({len(zero_weights)/len(all_weights)*100:.1f}%)")
    print(f"正常权重数: {len(non_zero_weights)} ({len(non_zero_weights)/len(all_weights)*100:.1f}%)")
    print()
    
    if len(zero_weights) > len(all_weights) * 0.5:
        print("⚠️  警告: 超过50%的权重全为0，这可能是:")
        print("  1. 模型文件损坏")
        print("  2. 模型文件未完整下载")
        print("  3. 模型文件被错误初始化")
        print()
        print("建议:")
        print("  1. 检查文件完整性（MD5/SHA256校验）")
        print("  2. 重新下载模型文件")
        print("  3. 联系模型提供者确认")
    
    print("=" * 80)


if __name__ == "__main__":
    main()



