import sys
from pathlib import Path
from safetensors import safe_open
import random

def main():
    """检查 model-00004-of-00004.safetensors 文件中的权重"""
    
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    weight_file = model_path / "model-00004-of-00004.safetensors"
    
    print("=" * 50)
    print("检查 model-00004-of-00004.safetensors 文件")
    print("=" * 50)
    
    with safe_open(str(weight_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"文件中的键数量: {len(keys)}")
        print()
        
        # 统计全为0的权重
        zero_weights = []
        non_zero_weights = []
        
        print("检查所有权重...")
        for key in keys:
            weight = f.get_tensor(key)
            non_zero = (weight != 0).sum().item()
            total = weight.numel()
            
            if non_zero == 0:
                zero_weights.append(key)
            else:
                non_zero_weights.append((key, non_zero, total))
        
        print()
        print("=" * 50)
        print("统计结果")
        print("=" * 50)
        print(f"全为 0 的权重数量: {len(zero_weights)}")
        print(f"包含非零值的权重数量: {len(non_zero_weights)}")
        print()
        
        if zero_weights:
            print("全为 0 的权重列表:")
            for key in zero_weights:
                print(f"  ✗ {key}")
            print()
        
        # 显示一些正常的权重
        if non_zero_weights:
            print("包含非零值的权重示例 (前10个):")
            for key, non_zero, total in non_zero_weights[:10]:
                weight = f.get_tensor(key)
                mean_val = weight.mean().item()
                print(f"  ✓ {key}: {non_zero}/{total} non-zero, mean={mean_val:.6f}")
            print()
        
        # 特别检查最后一层的相关权重
        print("=" * 50)
        print("最后一层相关权重检查")
        print("=" * 50)
        
        layer27_keys = [k for k in keys if "layers.27" in k or "norm.weight" in k]
        for key in sorted(layer27_keys):
            weight = f.get_tensor(key)
            non_zero = (weight != 0).sum().item()
            total = weight.numel()
            mean_val = weight.mean().item()
            
            status = "✓" if non_zero > 0 else "✗"
            print(f"{status} {key}:")
            print(f"    Non-zero: {non_zero}/{total}, Mean: {mean_val:.6f}")
            if non_zero == 0:
                print(f"    ⚠️  全为 0!")
            print()
    
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



