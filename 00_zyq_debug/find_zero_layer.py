import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """找到哪一层开始输出全0"""
    
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("找到零输出层")
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
    
    with torch.no_grad():
        base_model = model.get_model()
        inputs_embeds = base_model.embed_tokens(inputs.input_ids)
        position_ids, rope_deltas = model.get_rope_index(
            inputs.input_ids,
            None, None, None,
            inputs.attention_mask,
        )
        
        # 检查所有 hidden states
        base_outputs = base_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        print("检查所有层的 hidden states:")
        print("-" * 50)
        
        for i, hidden in enumerate(base_outputs.hidden_states):
            non_zero = (hidden != 0).sum().item()
            total = hidden.numel()
            mean_val = hidden.mean().item()
            std_val = hidden.std().item()
            
            status = "✓" if non_zero > 0 else "✗"
            print(f"{status} Layer {i:2d}: non-zero={non_zero:5d}/{total:5d}  mean={mean_val:8.6f}  std={std_val:8.6f}")
            
            # 找到第一个全0的层
            if non_zero == 0 and i > 0:
                prev_hidden = base_outputs.hidden_states[i-1]
                prev_non_zero = (prev_hidden != 0).sum().item()
                print(f"\n⚠️  发现：Layer {i} 开始全为0，但 Layer {i-1} 有 {prev_non_zero} 个非零值")
                print(f"   检查 Layer {i-1} 到 Layer {i} 之间的转换...")
                break
        
        print("-" * 50)
        print()
        
        # 检查模型结构
        print("检查模型结构:")
        print("-" * 50)
        if hasattr(base_model, 'layers'):
            print(f"Base model 有 {len(base_model.layers)} 层")
            for i, layer in enumerate(base_model.layers[:5]):  # 只显示前5层
                print(f"  Layer {i}: {type(layer).__name__}")
        print("-" * 50)
    
    print()
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



