import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """检查模型权重"""
    
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("检查模型权重")
    print("=" * 50)
    
    # 加载模型
    print("正在加载模型...")
    model = InternVLAN1ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    model.eval()
    print("✓ 模型加载成功")
    print()
    
    # 检查 lm_head
    print("=" * 50)
    print("检查 lm_head")
    print("=" * 50)
    
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        print(f"lm_head 类型: {type(lm_head)}")
        print(f"lm_head weight shape: {lm_head.weight.shape}")
        print(f"lm_head weight dtype: {lm_head.weight.dtype}")
        print(f"lm_head weight device: {lm_head.weight.device}")
        
        # 检查权重是否全为0
        weight_sum = lm_head.weight.sum().item()
        weight_mean = lm_head.weight.mean().item()
        weight_std = lm_head.weight.std().item()
        weight_max = lm_head.weight.max().item()
        weight_min = lm_head.weight.min().item()
        
        print(f"权重统计:")
        print(f"  Sum: {weight_sum:.6f}")
        print(f"  Mean: {weight_mean:.6f}")
        print(f"  Std: {weight_std:.6f}")
        print(f"  Max: {weight_max:.6f}")
        print(f"  Min: {weight_min:.6f}")
        print()
        
        # 检查前几个 token 的权重
        print("前5个 token 的权重（前5个维度）:")
        for token_id in range(5):
            token_weight = lm_head.weight[token_id, :5]
            print(f"  Token {token_id}: {token_weight.tolist()}")
        print()
    else:
        print("✗ 模型没有 lm_head")
        print()
    
    # 检查是否有其他输出层
    print("=" * 50)
    print("检查模型结构")
    print("=" * 50)
    
    # 查找所有 Linear 层
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'lm_head' in name or 'embed' in name or 'head' in name:
                linear_layers.append((name, module))
    
    print(f"找到 {len(linear_layers)} 个相关的 Linear 层:")
    for name, module in linear_layers[:10]:  # 只显示前10个
        print(f"  {name}: {module.weight.shape}")
    print()
    
    # 测试一个简单的 forward
    print("=" * 50)
    print("测试简单 Forward")
    print("=" * 50)
    
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    
    # 创建一个简单的输入
    test_text = "Hello"
    inputs = processor(text=[test_text], return_tensors="pt").to(device)
    
    print(f"输入文本: {test_text}")
    print(f"Input IDs: {inputs.input_ids[0].tolist()}")
    print()
    
    with torch.no_grad():
        # 只 forward 一次
        outputs = model(**inputs)
        logits = outputs.logits
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        
        # 检查 logits 的统计信息
        logits_sum = logits.sum().item()
        logits_mean = logits.mean().item()
        logits_std = logits.std().item()
        logits_max = logits.max().item()
        logits_min = logits.min().item()
        
        print(f"Logits 统计:")
        print(f"  Sum: {logits_sum:.6f}")
        print(f"  Mean: {logits_mean:.6f}")
        print(f"  Std: {logits_std:.6f}")
        print(f"  Max: {logits_max:.6f}")
        print(f"  Min: {logits_min:.6f}")
        print()
        
        # 检查最后一个位置的 logits
        last_logits = logits[0, -1, :]
        last_logits_sum = last_logits.sum().item()
        last_logits_mean = last_logits.mean().item()
        last_logits_std = last_logits.std().item()
        
        print(f"最后一个位置的 Logits 统计:")
        print(f"  Sum: {last_logits_sum:.6f}")
        print(f"  Mean: {last_logits_mean:.6f}")
        print(f"  Std: {last_logits_std:.6f}")
        print()
        
        # 检查是否有非零值
        non_zero_count = (last_logits != 0).sum().item()
        print(f"非零 logits 数量: {non_zero_count} / {len(last_logits)}")
        print()
    
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



