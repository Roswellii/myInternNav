#!/usr/bin/env python3
"""
检查 InternVLA-N1-wo-dagger 模型权重文件中的零值统计
"""

import sys
from pathlib import Path
from safetensors import safe_open
import torch

def check_weight_file(weight_file, sample_keys=None):
    """检查单个权重文件中的零值情况"""
    print(f"\n{'='*80}")
    print(f"检查文件: {weight_file.name}")
    print(f"{'='*80}")
    
    results = {
        'total_keys': 0,
        'zero_keys': [],
        'partial_zero_keys': [],
        'normal_keys': [],
        'total_params': 0,
        'zero_params': 0,
    }
    
    try:
        with safe_open(str(weight_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            results['total_keys'] = len(keys)
            
            print(f"权重键数量: {len(keys)}")
            print(f"开始检查...")
            
            # 如果指定了采样，只检查部分键
            if sample_keys is not None:
                keys_to_check = [k for k in keys if any(sk in k for sk in sample_keys)]
                if not keys_to_check:
                    keys_to_check = keys[:min(20, len(keys))]  # 至少检查前20个
            else:
                keys_to_check = keys
            
            for idx, key in enumerate(keys_to_check):
                if idx % 50 == 0 and idx > 0:
                    print(f"  已检查 {idx}/{len(keys_to_check)} 个权重...")
                
                try:
                    weight = f.get_tensor(key)
                    total = weight.numel()
                    results['total_params'] += total
                    
                    # 统计零值
                    zero_count = (weight == 0).sum().item()
                    non_zero_count = total - zero_count
                    zero_ratio = zero_count / total if total > 0 else 0
                    
                    # 计算统计信息
                    mean_val = weight.float().mean().item()
                    std_val = weight.float().std().item()
                    min_val = weight.float().min().item()
                    max_val = weight.float().max().item()
                    
                    results['zero_params'] += zero_count
                    
                    # 分类
                    if zero_count == total:
                        results['zero_keys'].append({
                            'key': key,
                            'shape': list(weight.shape),
                            'total': total,
                            'zero_ratio': 1.0
                        })
                    elif zero_ratio > 0.5:  # 超过50%为零
                        results['partial_zero_keys'].append({
                            'key': key,
                            'shape': list(weight.shape),
                            'total': total,
                            'zero_count': zero_count,
                            'zero_ratio': zero_ratio,
                            'mean': mean_val,
                            'std': std_val,
                        })
                    else:
                        results['normal_keys'].append({
                            'key': key,
                            'shape': list(weight.shape),
                            'total': total,
                            'zero_count': zero_count,
                            'zero_ratio': zero_ratio,
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                        })
                        
                except Exception as e:
                    print(f"  ⚠️  检查 {key} 时出错: {e}")
                    continue
            
            # 如果只检查了部分键，估算总体情况
            if sample_keys is not None and len(keys_to_check) < len(keys):
                scale_factor = len(keys) / len(keys_to_check)
                print(f"\n  注意: 只检查了 {len(keys_to_check)}/{len(keys)} 个权重键")
                print(f"  以下统计基于采样结果")
    
    except Exception as e:
        print(f"❌ 打开文件失败: {e}")
        return None
    
    return results

def print_results(results, file_name):
    """打印检查结果"""
    if results is None:
        return
    
    print(f"\n{'='*80}")
    print(f"统计结果: {file_name}")
    print(f"{'='*80}")
    
    print(f"\n总体统计:")
    print(f"  权重键总数: {results['total_keys']}")
    print(f"  参数总数: {results['total_params']:,}")
    print(f"  零值参数数: {results['zero_params']:,}")
    if results['total_params'] > 0:
        zero_ratio = results['zero_params'] / results['total_params']
        print(f"  零值比例: {zero_ratio*100:.2f}%")
    
    print(f"\n分类统计:")
    print(f"  全为0的权重: {len(results['zero_keys'])}")
    print(f"  零值>50%的权重: {len(results['partial_zero_keys'])}")
    print(f"  正常权重: {len(results['normal_keys'])}")
    
    # 显示全为0的权重
    if results['zero_keys']:
        print(f"\n⚠️  全为0的权重列表 (前20个):")
        for item in results['zero_keys'][:20]:
            print(f"  ✗ {item['key']} (shape: {item['shape']})")
        if len(results['zero_keys']) > 20:
            print(f"  ... 还有 {len(results['zero_keys']) - 20} 个全为0的权重")
    
    # 显示零值较多的权重
    if results['partial_zero_keys']:
        print(f"\n⚠️  零值>50%的权重 (前10个):")
        sorted_partial = sorted(results['partial_zero_keys'], 
                               key=lambda x: x['zero_ratio'], reverse=True)
        for item in sorted_partial[:10]:
            print(f"  ⚠️  {item['key']}: {item['zero_ratio']*100:.1f}% 为零 "
                  f"(shape: {item['shape']}, mean: {item['mean']:.6f})")
    
    # 显示一些正常权重的统计
    if results['normal_keys']:
        print(f"\n✓ 正常权重示例 (前5个):")
        for item in results['normal_keys'][:5]:
            print(f"  ✓ {item['key']}: "
                  f"零值比例 {item['zero_ratio']*100:.2f}%, "
                  f"mean={item['mean']:.6f}, std={item['std']:.6f}, "
                  f"range=[{item['min']:.4f}, {item['max']:.4f}]")

def main():
    """主函数"""
    model_dir = Path(__file__).parent / "checkpoints" / "InternVLA-N1-wo-dagger"
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    print("="*80)
    print("检查 InternVLA-N1-wo-dagger 模型权重文件中的零值")
    print("="*80)
    
    # 检查所有权重文件
    weight_files = sorted(model_dir.glob("model-*.safetensors"))
    
    if not weight_files:
        print(f"❌ 未找到权重文件: {model_dir}")
        sys.exit(1)
    
    print(f"\n找到 {len(weight_files)} 个权重文件")
    
    all_results = {}
    
    for weight_file in weight_files:
        # 对于大文件，可以采样检查关键层
        # 检查所有层，但可以限制每个文件的键数量
        results = check_weight_file(weight_file)
        if results:
            all_results[weight_file.name] = results
            print_results(results, weight_file.name)
    
    # 汇总统计
    print(f"\n\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    
    total_params = sum(r['total_params'] for r in all_results.values())
    total_zeros = sum(r['zero_params'] for r in all_results.values())
    total_zero_keys = sum(len(r['zero_keys']) for r in all_results.values())
    total_partial_zero_keys = sum(len(r['partial_zero_keys']) for r in all_results.values())
    
    print(f"\n所有文件汇总:")
    print(f"  总参数数: {total_params:,}")
    print(f"  总零值参数数: {total_zeros:,}")
    if total_params > 0:
        overall_zero_ratio = total_zeros / total_params
        print(f"  总体零值比例: {overall_zero_ratio*100:.2f}%")
    
    print(f"\n  全为0的权重总数: {total_zero_keys}")
    print(f"  零值>50%的权重总数: {total_partial_zero_keys}")
    
    if overall_zero_ratio > 0.1:
        print(f"\n⚠️  警告: 零值比例较高 ({overall_zero_ratio*100:.2f}%)")
        print(f"  这可能表示:")
        print(f"    1. 某些层未正确初始化")
        print(f"    2. 下载不完整或文件损坏")
        print(f"    3. 模型使用了稀疏权重")
    else:
        print(f"\n✓ 零值比例正常 ({overall_zero_ratio*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print("检查完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

