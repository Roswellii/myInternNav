#!/usr/bin/env python
"""
专门评估 model-00004-of-00004.safetensors 文件
深入分析权重分布、关键层状态和潜在问题
"""
import sys
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict
from datetime import datetime
import json

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
    
    # 计算零值的分布
    zero_count = total - non_zero
    zero_ratio = zero_count / total if total > 0 else 0
    
    return {
        'key': key,
        'shape': list(weight.shape),
        'dtype': str(weight.dtype),
        'total': total,
        'non_zero': non_zero,
        'zero_count': zero_count,
        'zero_ratio': zero_ratio,
        'mean': mean_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'sum': sum_val,
        'is_all_zero': non_zero == 0,
        'is_all_non_zero': zero_count == 0,
    }


def evaluate_file(file_path):
    """评估单个文件"""
    print("=" * 80)
    print("评估文件: model-00004-of-00004.safetensors")
    print("=" * 80)
    print(f"文件路径: {file_path}")
    print(f"文件存在: {file_path.exists()}")
    
    if not file_path.exists():
        print("✗ 文件不存在!")
        return None
    
    # 文件基本信息
    file_size = file_path.stat().st_size
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    print()
    
    # 读取并分析所有权重
    print("正在读取和分析权重...")
    all_weights = []
    zero_weights = []
    non_zero_weights = []
    
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"总权重数量: {len(keys)}")
        print()
        
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
        print()
    
    # 统计摘要
    print("=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"总权重数量: {len(all_weights)}")
    print(f"全为0的权重: {len(zero_weights)} ({len(zero_weights)/len(all_weights)*100:.2f}%)")
    print(f"包含非零值的权重: {len(non_zero_weights)} ({len(non_zero_weights)/len(all_weights)*100:.2f}%)")
    print()
    
    # 按模块分类
    print("=" * 80)
    print("按模块分类统计")
    print("=" * 80)
    
    module_stats = defaultdict(lambda: {
        'total': 0, 
        'zero': 0, 
        'non_zero': 0,
        'total_params': 0,
        'zero_params': 0,
        'non_zero_params': 0,
    })
    
    for stats in all_weights:
        parts = stats['key'].split('.')
        if len(parts) >= 2:
            module = '.'.join(parts[:2])
        else:
            module = parts[0]
        
        module_stats[module]['total'] += 1
        module_stats[module]['total_params'] += stats['total']
        
        if stats['is_all_zero']:
            module_stats[module]['zero'] += 1
            module_stats[module]['zero_params'] += stats['total']
        else:
            module_stats[module]['non_zero'] += 1
            module_stats[module]['non_zero_params'] += stats['total']
    
    for module in sorted(module_stats.keys()):
        stats = module_stats[module]
        zero_ratio = stats['zero'] / stats['total'] * 100 if stats['total'] > 0 else 0
        param_zero_ratio = stats['zero_params'] / stats['total_params'] * 100 if stats['total_params'] > 0 else 0
        status = "⚠️" if zero_ratio > 50 else "✓"
        print(f"{status} {module}:")
        print(f"    权重数量: 总数={stats['total']}, 全0={stats['zero']} ({zero_ratio:.1f}%), 正常={stats['non_zero']}")
        print(f"    参数数量: 总数={stats['total_params']:,}, 全0={stats['zero_params']:,} ({param_zero_ratio:.1f}%), 正常={stats['non_zero_params']:,}")
    print()
    
    # 关键层详细检查
    print("=" * 80)
    print("关键层详细检查")
    print("=" * 80)
    
    critical_keys = [
        "model.norm.weight",
        "model.layers.26.input_layernorm.weight",
        "model.layers.26.post_attention_layernorm.weight",
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
    
    critical_results = {}
    
    for key in critical_keys:
        found = False
        for stats in all_weights:
            if stats['key'] == key:
                found = True
                critical_results[key] = stats
                status = "✗" if stats['is_all_zero'] else "✓"
                print(f"{status} {key}:")
                print(f"    Shape: {stats['shape']}")
                print(f"    Total params: {stats['total']:,}")
                print(f"    Non-zero: {stats['non_zero']:,} / {stats['total']:,} ({stats['non_zero']/stats['total']*100:.2f}%)")
                print(f"    Zero: {stats['zero_count']:,} / {stats['total']:,} ({stats['zero_ratio']*100:.2f}%)")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Std: {stats['std']:.6f}")
                print(f"    Max: {stats['max']:.6f}, Min: {stats['min']:.6f}")
                print(f"    Sum: {stats['sum']:.6f}")
                if stats['is_all_zero']:
                    print(f"    ⚠️  全为 0 - 这是严重问题!")
                print()
                break
        if not found:
            print(f"✗ {key}: 未找到")
            print()
    
    # Layer 27 完整检查
    print("=" * 80)
    print("Layer 27 完整检查")
    print("=" * 80)
    
    layer27_weights = [stats for stats in all_weights if 'layers.27' in stats['key']]
    if layer27_weights:
        print(f"找到 {len(layer27_weights)} 个 Layer 27 的权重:")
        print()
        for stats in sorted(layer27_weights, key=lambda x: x['key']):
            status = "✗" if stats['is_all_zero'] else "✓"
            print(f"  {status} {stats['key']}:")
            print(f"    Shape: {stats['shape']}, Params: {stats['total']:,}")
            print(f"    Non-zero: {stats['non_zero']:,}/{stats['total']:,} ({stats['non_zero']/stats['total']*100:.2f}%)")
            if stats['is_all_zero']:
                print(f"    ⚠️  全为 0!")
            print()
    else:
        print("未找到 Layer 27 的权重")
        print()
    
    # NavDP 模块检查
    print("=" * 80)
    print("NavDP 模块检查")
    print("=" * 80)
    
    navdp_weights = [stats for stats in all_weights if 'navdp' in stats['key']]
    if navdp_weights:
        navdp_zero = sum(1 for stats in navdp_weights if stats['is_all_zero'])
        navdp_total_params = sum(stats['total'] for stats in navdp_weights)
        navdp_zero_params = sum(stats['total'] for stats in navdp_weights if stats['is_all_zero'])
        
        print(f"NavDP 权重数量: {len(navdp_weights)}")
        print(f"全为0的权重: {navdp_zero} ({navdp_zero/len(navdp_weights)*100:.1f}%)")
        print(f"总参数数: {navdp_total_params:,}")
        print(f"全为0的参数: {navdp_zero_params:,} ({navdp_zero_params/navdp_total_params*100:.1f}%)")
        print()
        
        # 按子模块分类
        navdp_modules = defaultdict(list)
        for stats in navdp_weights:
            parts = stats['key'].split('.')
            if len(parts) >= 3:
                submodule = '.'.join(parts[2:4])  # 例如 "decoder.layers"
            else:
                submodule = parts[-1]
            navdp_modules[submodule].append(stats)
        
        print("NavDP 子模块统计:")
        for submodule in sorted(navdp_modules.keys())[:10]:  # 只显示前10个
            weights = navdp_modules[submodule]
            zero_count = sum(1 for w in weights if w['is_all_zero'])
            print(f"  {submodule}: {len(weights)} 个权重, {zero_count} 个全0 ({zero_count/len(weights)*100:.1f}%)")
        print()
    else:
        print("未找到 NavDP 权重")
        print()
    
    # 正常权重的详细信息
    print("=" * 80)
    print("包含非零值的权重详情")
    print("=" * 80)
    
    if non_zero_weights:
        for stats in non_zero_weights:
            print(f"✓ {stats['key']}:")
            print(f"    Shape: {stats['shape']}")
            print(f"    Total params: {stats['total']:,}")
            print(f"    Non-zero: {stats['non_zero']:,} ({stats['non_zero']/stats['total']*100:.2f}%)")
            print(f"    Zero: {stats['zero_count']:,} ({stats['zero_ratio']*100:.2f}%)")
            print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"    Max: {stats['max']:.6f}, Min: {stats['min']:.6f}")
            print()
    else:
        print("✗ 没有找到包含非零值的权重!")
        print()
    
    # 问题影响分析
    print("=" * 80)
    print("问题影响分析")
    print("=" * 80)
    
    issues = []
    
    # 检查 model.norm.weight
    norm_weight = next((stats for stats in all_weights if stats['key'] == 'model.norm.weight'), None)
    if norm_weight and norm_weight['is_all_zero']:
        issues.append({
            'severity': 'CRITICAL',
            'layer': 'model.norm.weight',
            'impact': '导致模型最后一层输出全为0，进而导致所有logits全为0',
            'effect': '模型无法生成有效文本，总是输出token 0'
        })
    
    # 检查 Layer 27
    layer27_all_zero = all(stats['is_all_zero'] for stats in layer27_weights)
    if layer27_all_zero and layer27_weights:
        issues.append({
            'severity': 'CRITICAL',
            'layer': 'model.layers.27.*',
            'impact': '最后一层transformer层完全失效',
            'effect': '模型无法进行有效的特征提取和转换'
        })
    
    # 检查 NavDP
    navdp_all_zero = all(stats['is_all_zero'] for stats in navdp_weights) if navdp_weights else False
    if navdp_all_zero:
        issues.append({
            'severity': 'HIGH',
            'layer': 'model.navdp.*',
            'impact': '导航决策模块完全失效',
            'effect': '无法进行导航相关的推理和决策'
        })
    
    if issues:
        print("发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. [{issue['severity']}] {issue['layer']}")
            print(f"   影响: {issue['impact']}")
            print(f"   效果: {issue['effect']}")
    else:
        print("✓ 未发现严重问题")
    print()
    
    # 生成评估报告
    print("=" * 80)
    print("生成评估报告...")
    print("=" * 80)
    
    report = {
        'file_name': file_path.name,
        'file_path': str(file_path),
        'file_size_gb': file_size / (1024**3),
        'evaluation_time': datetime.now().isoformat(),
        'summary': {
            'total_weights': len(all_weights),
            'zero_weights': len(zero_weights),
            'non_zero_weights': len(non_zero_weights),
            'zero_ratio': len(zero_weights) / len(all_weights) if all_weights else 0,
        },
        'module_stats': {
            k: {
                'total': v['total'],
                'zero': v['zero'],
                'non_zero': v['non_zero'],
                'total_params': v['total_params'],
                'zero_params': v['zero_params'],
                'non_zero_params': v['non_zero_params'],
            }
            for k, v in module_stats.items()
        },
        'critical_layers': {
            k: {
                'shape': v['shape'],
                'total': v['total'],
                'non_zero': v['non_zero'],
                'zero_count': v['zero_count'],
                'is_all_zero': v['is_all_zero'],
                'mean': v['mean'],
                'std': v['std'],
            }
            for k, v in critical_results.items()
        },
        'issues': issues,
    }
    
    # 保存 JSON 报告（保存到文件所在目录）
    report_file_json = file_path.parent / "file4_evaluation_report.json"
    with open(report_file_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON 报告已保存到: {report_file_json}")
    
    # 保存文本报告（保存到文件所在目录）
    report_file_txt = file_path.parent / "file4_evaluation_report.txt"
    with open(report_file_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("model-00004-of-00004.safetensors 评估报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件大小: {file_size / (1024**3):.2f} GB\n")
        f.write("\n")
        f.write("统计摘要:\n")
        f.write(f"  总权重数: {len(all_weights)}\n")
        f.write(f"  全为0的权重: {len(zero_weights)} ({len(zero_weights)/len(all_weights)*100:.2f}%)\n")
        f.write(f"  正常权重: {len(non_zero_weights)} ({len(non_zero_weights)/len(all_weights)*100:.2f}%)\n")
        f.write("\n")
        f.write("关键层状态:\n")
        for key, stats in critical_results.items():
            status = "✗ 全为0" if stats['is_all_zero'] else "✓ 正常"
            f.write(f"  {key}: {status}\n")
        f.write("\n")
        f.write("问题列表:\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"  {i}. [{issue['severity']}] {issue['layer']}\n")
            f.write(f"     影响: {issue['impact']}\n")
            f.write(f"     效果: {issue['effect']}\n")
    
    print(f"✓ 文本报告已保存到: {report_file_txt}")
    print()
    
    # 最终评估
    print("=" * 80)
    print("最终评估")
    print("=" * 80)
    
    zero_ratio = len(zero_weights) / len(all_weights) if all_weights else 0
    
    if zero_ratio > 0.99:
        print("⚠️  严重问题: 超过 99% 的权重全为 0")
        print("   评估: 文件存在严重问题，模型无法正常工作")
        print("   建议: 立即重新下载此文件")
    elif zero_ratio > 0.9:
        print("⚠️  重大问题: 超过 90% 的权重全为 0")
        print("   评估: 文件存在重大问题，模型功能严重受限")
        print("   建议: 重新下载此文件")
    elif zero_ratio > 0.5:
        print("⚠️  问题: 超过 50% 的权重全为 0")
        print("   评估: 文件存在问题，模型功能受限")
        print("   建议: 检查文件完整性，考虑重新下载")
    else:
        print("✓ 文件状态正常")
    
    print()
    print("=" * 80)
    
    return report


def main():
    """主函数"""
    # 评估 InternVLA-N1-wo-dagger 目录下的文件
    file_path = Path("/home/zhangwenqi/workspace/myInternNav/checkpoints/InternVLA-N1-wo-dagger/model-00004-of-00004.safetensors")
    
    try:
        report = evaluate_file(file_path)
        if report:
            print("\n评估完成!")
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

