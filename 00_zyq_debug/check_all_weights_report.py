#!/usr/bin/env python
"""
检查所有模型文件中的权重并生成详细报告
"""
import sys
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict
from datetime import datetime

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


def check_file(file_path, file_name):
    """检查单个文件"""
    print(f"\n{'=' * 80}")
    print(f"检查文件: {file_name}")
    print(f"{'=' * 80}")
    
    all_weights = []
    zero_weights = []
    non_zero_weights = []
    
    try:
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"总权重数量: {len(keys)}")
            
            for i, key in enumerate(keys):
                weight = f.get_tensor(key)
                stats = analyze_weight(weight, key)
                all_weights.append(stats)
                
                if stats['is_all_zero']:
                    zero_weights.append(stats)
                else:
                    non_zero_weights.append(stats)
                
                if (i + 1) % 100 == 0:
                    print(f"  已分析 {i + 1}/{len(keys)} 个权重...")
            
            print(f"✓ 分析完成")
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return None
    
    return {
        'file_name': file_name,
        'file_path': str(file_path),
        'total_weights': len(all_weights),
        'zero_weights': zero_weights,
        'non_zero_weights': non_zero_weights,
        'all_weights': all_weights,
    }


def generate_report(results):
    """生成报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("模型权重检查报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 总体统计
    report_lines.append("=" * 80)
    report_lines.append("总体统计")
    report_lines.append("=" * 80)
    
    total_files = len(results)
    total_weights = sum(r['total_weights'] for r in results)
    total_zero = sum(len(r['zero_weights']) for r in results)
    total_non_zero = sum(len(r['non_zero_weights']) for r in results)
    
    report_lines.append(f"检查的文件数: {total_files}")
    report_lines.append(f"总权重数量: {total_weights}")
    report_lines.append(f"全为0的权重: {total_zero} ({total_zero/total_weights*100:.2f}%)")
    report_lines.append(f"正常权重: {total_non_zero} ({total_non_zero/total_weights*100:.2f}%)")
    report_lines.append("")
    
    # 每个文件的详细统计
    for result in results:
        report_lines.append("=" * 80)
        report_lines.append(f"文件: {result['file_name']}")
        report_lines.append("=" * 80)
        report_lines.append(f"总权重数: {result['total_weights']}")
        report_lines.append(f"全为0的权重: {len(result['zero_weights'])} ({len(result['zero_weights'])/result['total_weights']*100:.2f}%)")
        report_lines.append(f"正常权重: {len(result['non_zero_weights'])} ({len(result['non_zero_weights'])/result['total_weights']*100:.2f}%)")
        report_lines.append("")
        
        # 按模块分类
        module_stats = defaultdict(lambda: {'total': 0, 'zero': 0, 'non_zero': 0})
        
        for stats in result['all_weights']:
            parts = stats['key'].split('.')
            if len(parts) >= 2:
                module = '.'.join(parts[:2])
            else:
                module = parts[0]
            
            module_stats[module]['total'] += 1
            if stats['is_all_zero']:
                module_stats[module]['zero'] += 1
            else:
                module_stats[module]['non_zero'] += 1
        
        report_lines.append("按模块分类:")
        for module in sorted(module_stats.keys()):
            stats = module_stats[module]
            zero_ratio = stats['zero'] / stats['total'] * 100 if stats['total'] > 0 else 0
            status = "⚠️" if zero_ratio > 50 else "✓"
            report_lines.append(f"  {status} {module}: 总数={stats['total']}, 全0={stats['zero']} ({zero_ratio:.1f}%), 正常={stats['non_zero']}")
        report_lines.append("")
        
        # 关键层检查
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
        
        report_lines.append("关键层检查:")
        for key in critical_keys:
            found = False
            for stats in result['all_weights']:
                if stats['key'] == key:
                    found = True
                    status = "✗" if stats['is_all_zero'] else "✓"
                    report_lines.append(f"  {status} {key}:")
                    report_lines.append(f"    Shape: {stats['shape']}")
                    report_lines.append(f"    Non-zero: {stats['non_zero']} / {stats['total']}")
                    report_lines.append(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                    if stats['is_all_zero']:
                        report_lines.append(f"    ⚠️  全为 0!")
                    break
            if not found:
                report_lines.append(f"  ✗ {key}: 未找到")
        report_lines.append("")
        
        # 全为0的权重列表（只显示前20个）
        if result['zero_weights']:
            report_lines.append(f"全为0的权重 (显示前20个，共{len(result['zero_weights'])}个):")
            for stats in result['zero_weights'][:20]:
                report_lines.append(f"  ✗ {stats['key']}: shape={stats['shape']}")
            if len(result['zero_weights']) > 20:
                report_lines.append(f"  ... 还有 {len(result['zero_weights']) - 20} 个全0权重")
            report_lines.append("")
    
    # 问题总结
    report_lines.append("=" * 80)
    report_lines.append("问题总结")
    report_lines.append("=" * 80)
    
    # 找出有问题的文件
    problematic_files = []
    for result in results:
        zero_ratio = len(result['zero_weights']) / result['total_weights'] if result['total_weights'] > 0 else 0
        if zero_ratio > 0.5:
            problematic_files.append((result['file_name'], zero_ratio))
    
    if problematic_files:
        report_lines.append("⚠️  发现问题的文件:")
        for file_name, zero_ratio in problematic_files:
            report_lines.append(f"  - {file_name}: {zero_ratio*100:.1f}% 的权重全为0")
        report_lines.append("")
        report_lines.append("可能的原因:")
        report_lines.append("  1. 模型文件损坏")
        report_lines.append("  2. 模型文件未完整下载")
        report_lines.append("  3. 模型文件被错误初始化")
        report_lines.append("  4. 模型保存时出现问题")
        report_lines.append("")
        report_lines.append("建议:")
        report_lines.append("  1. 检查文件完整性（MD5/SHA256校验）")
        report_lines.append("  2. 重新下载有问题的文件")
        report_lines.append("  3. 联系模型提供者确认")
    else:
        report_lines.append("✓ 所有文件的权重检查正常")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    
    print("=" * 80)
    print("模型权重检查 - 生成详细报告")
    print("=" * 80)
    print(f"模型目录: {model_path}")
    print()
    
    # 查找所有 safetensors 文件
    safetensors_files = sorted(model_path.glob("model-*.safetensors"))
    
    if not safetensors_files:
        print("✗ 未找到 safetensors 文件!")
        return
    
    print(f"找到 {len(safetensors_files)} 个模型文件")
    print()
    
    # 检查每个文件
    results = []
    for file_path in safetensors_files:
        result = check_file(file_path, file_path.name)
        if result:
            results.append(result)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("生成报告...")
    print("=" * 80)
    
    report = generate_report(results)
    
    # 保存报告
    report_file = model_path / "weights_check_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 报告已保存到: {report_file}")
    print()
    
    # 显示报告摘要
    print("=" * 80)
    print("报告摘要")
    print("=" * 80)
    print()
    
    # 只显示关键信息
    total_weights = sum(r['total_weights'] for r in results)
    total_zero = sum(len(r['zero_weights']) for r in results)
    
    print(f"总权重数: {total_weights}")
    print(f"全为0的权重: {total_zero} ({total_zero/total_weights*100:.2f}%)")
    print()
    
    for result in results:
        zero_ratio = len(result['zero_weights']) / result['total_weights'] if result['total_weights'] > 0 else 0
        status = "⚠️" if zero_ratio > 0.5 else "✓"
        print(f"{status} {result['file_name']}: {len(result['zero_weights'])}/{result['total_weights']} 全为0 ({zero_ratio*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("详细报告已保存到: weights_check_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



