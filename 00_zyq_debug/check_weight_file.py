import sys
from pathlib import Path
import torch
from safetensors import safe_open

sys.path.append(str(Path(__file__).parent.parent))

def main():
    """检查权重文件中的 norm.weight 值"""
    
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    weight_file = model_path / "model-00004-of-00004.safetensors"
    
    print("=" * 50)
    print("检查权重文件中的 model.norm.weight")
    print("=" * 50)
    print(f"权重文件: {weight_file}")
    print(f"文件存在: {weight_file.exists()}")
    print()
    
    if not weight_file.exists():
        print("✗ 文件不存在!")
        return
    
    # 打开 safetensors 文件
    print("正在读取权重文件...")
    with safe_open(str(weight_file), framework="pt", device="cpu") as f:
        # 列出所有键
        keys = f.keys()
        print(f"文件中的键数量: {len(list(keys))}")
        print()
        
        # 检查是否有 model.norm.weight
        if "model.norm.weight" in keys:
            print("✓ 找到 model.norm.weight")
            print()
            
            # 读取权重
            norm_weight = f.get_tensor("model.norm.weight")
            
            print("权重信息:")
            print(f"  Shape: {norm_weight.shape}")
            print(f"  Dtype: {norm_weight.dtype}")
            print()
            
            # 统计信息
            weight_sum = norm_weight.sum().item()
            weight_mean = norm_weight.mean().item()
            weight_std = norm_weight.std().item()
            weight_max = norm_weight.max().item()
            weight_min = norm_weight.min().item()
            non_zero_count = (norm_weight != 0).sum().item()
            total_count = norm_weight.numel()
            
            print("权重统计:")
            print(f"  Sum: {weight_sum:.6f}")
            print(f"  Mean: {weight_mean:.6f}")
            print(f"  Std: {weight_std:.6f}")
            print(f"  Max: {weight_max:.6f}")
            print(f"  Min: {weight_min:.6f}")
            print(f"  Non-zero: {non_zero_count} / {total_count}")
            print()
            
            # 显示前20个值
            print("前20个权重值:")
            values = norm_weight[:20].tolist()
            for i, val in enumerate(values):
                print(f"  [{i:2d}]: {val:.6f}")
            print()
            
            # 检查是否全为0
            if non_zero_count == 0:
                print("⚠️  警告: 权重全为 0!")
            else:
                print("✓ 权重包含非零值")
        else:
            print("✗ 未找到 model.norm.weight")
            print()
            print("文件中包含的 norm 相关键:")
            norm_keys = [k for k in keys if "norm" in k.lower()]
            for key in sorted(norm_keys)[:20]:  # 只显示前20个
                print(f"  {key}")
    
    print()
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



