import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """检查 hidden states"""
    
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("检查 Hidden States")
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
    
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    
    # 测试输入
    test_text = "Hello"
    inputs = processor(text=[test_text], return_tensors="pt").to(device)
    
    print(f"输入文本: {test_text}")
    print(f"Input IDs: {inputs.input_ids[0].tolist()}")
    print()
    
    with torch.no_grad():
        # Forward with output_hidden_states
        print("=" * 50)
        print("Forward with output_hidden_states=True")
        print("=" * 50)
        
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        
        print(f"Logits shape: {logits.shape}")
        print(f"Hidden states 数量: {len(hidden_states)}")
        print()
        
        # 检查最后一个 hidden state
        if hidden_states:
            last_hidden = hidden_states[-1]
            print(f"最后一个 hidden state shape: {last_hidden.shape}")
            print(f"最后一个 hidden state dtype: {last_hidden.dtype}")
            
            # 统计信息
            hidden_sum = last_hidden.sum().item()
            hidden_mean = last_hidden.mean().item()
            hidden_std = last_hidden.std().item()
            hidden_max = last_hidden.max().item()
            hidden_min = last_hidden.min().item()
            non_zero_count = (last_hidden != 0).sum().item()
            
            print(f"最后一个 hidden state 统计:")
            print(f"  Sum: {hidden_sum:.6f}")
            print(f"  Mean: {hidden_mean:.6f}")
            print(f"  Std: {hidden_std:.6f}")
            print(f"  Max: {hidden_max:.6f}")
            print(f"  Min: {hidden_min:.6f}")
            print(f"  非零值数量: {non_zero_count} / {last_hidden.numel()}")
            print()
            
            # 检查最后一个位置的 hidden state
            last_pos_hidden = last_hidden[0, -1, :]
            print(f"最后一个位置的 hidden state (前10个维度):")
            print(f"  {last_pos_hidden[:10].tolist()}")
            print()
            
            # 手动计算 logits
            print("=" * 50)
            print("手动计算 Logits")
            print("=" * 50)
            
            last_pos_hidden_float = last_pos_hidden.float()
            lm_head_weight = model.lm_head.weight.float()
            
            # 计算 logits
            manual_logits = torch.matmul(last_pos_hidden_float, lm_head_weight.t())
            
            print(f"手动计算的 logits shape: {manual_logits.shape}")
            print(f"手动计算的 logits 统计:")
            print(f"  Sum: {manual_logits.sum().item():.6f}")
            print(f"  Mean: {manual_logits.mean().item():.6f}")
            print(f"  Std: {manual_logits.std().item():.6f}")
            print(f"  Max: {manual_logits.max().item():.6f}")
            print(f"  Min: {manual_logits.min().item():.6f}")
            print()
            
            # 比较模型输出的 logits
            model_logits = logits[0, -1, :].float()
            print(f"模型输出的 logits 统计:")
            print(f"  Sum: {model_logits.sum().item():.6f}")
            print(f"  Mean: {model_logits.mean().item():.6f}")
            print(f"  Std: {model_logits.std().item():.6f}")
            print()
            
            # 检查差异
            diff = (manual_logits - model_logits).abs()
            print(f"差异统计:")
            print(f"  Max diff: {diff.max().item():.6f}")
            print(f"  Mean diff: {diff.mean().item():.6f}")
            print()
    
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



