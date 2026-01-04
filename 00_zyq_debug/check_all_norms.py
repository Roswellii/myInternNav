import sys
from pathlib import Path
from safetensors import safe_open

def main():
    """检查所有 norm 层的权重"""
    
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    
    # 根据索引文件，norm 权重分布在多个文件中
    weight_files = {
        "model-00001-of-00004.safetensors": ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight"],
        "model-00004-of-00004.safetensors": ["model.norm.weight", "model.layers.27.input_layernorm.weight", "model.layers.27.post_attention_layernorm.weight"],
    }
    
    print("=" * 50)
    print("检查所有 Norm 层权重")
    print("=" * 50)
    print()
    
    for weight_file, keys_to_check in weight_files.items():
        file_path = model_path / weight_file
        print(f"文件: {weight_file}")
        print("-" * 50)
        
        if not file_path.exists():
            print("  ✗ 文件不存在!")
            print()
            continue
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in keys_to_check:
                if key in f.keys():
                    weight = f.get_tensor(key)
                    non_zero = (weight != 0).sum().item()
                    total = weight.numel()
                    mean_val = weight.mean().item()
                    std_val = weight.std().item()
                    
                    status = "✓" if non_zero > 0 else "✗"
                    print(f"  {status} {key}:")
                    print(f"    Shape: {weight.shape}")
                    print(f"    Non-zero: {non_zero} / {total}")
                    print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                    if non_zero == 0:
                        print(f"    ⚠️  全为 0!")
                else:
                    print(f"  ✗ {key}: 未找到")
                print()
    
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)
    
    # 总结
    print()
    print("=" * 50)
    print("问题总结")
    print("=" * 50)
    print("问题层: model.norm.weight")
    print("位置: model-00004-of-00004.safetensors")
    print("状态: 权重全为 0 (3584 个值全部为 0)")
    print()
    print("影响: 导致模型最后一层输出全为 0，进而导致 logits 全为 0")
    print("结果: 模型生成时总是选择 token 0 ('!')")
    print()
    print("建议: 这是一个模型权重文件的问题，需要:")
    print("  1. 检查模型文件是否完整/损坏")
    print("  2. 重新下载模型文件")
    print("  3. 联系模型提供者确认这是否是已知问题")


if __name__ == "__main__":
    main()



