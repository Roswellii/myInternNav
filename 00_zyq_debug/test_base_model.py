import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """测试 base model 直接调用"""
    
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("测试 Base Model 直接调用")
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
        # 获取 base model
        base_model = model.get_model()
        
        # 获取 embeddings
        inputs_embeds = base_model.embed_tokens(inputs.input_ids)
        print(f"Input embeddings shape: {inputs_embeds.shape}")
        print(f"Input embeddings non-zero: {(inputs_embeds != 0).sum().item()} / {inputs_embeds.numel()}")
        print()
        
        # 获取 position_ids
        position_ids, rope_deltas = model.get_rope_index(
            inputs.input_ids,
            None,  # image_grid_thw
            None,  # video_grid_thw
            None,  # second_per_grid_ts
            inputs.attention_mask,
        )
        print(f"Position IDs shape: {position_ids.shape}")
        print(f"Position IDs: {position_ids}")
        print()
        
        # 直接调用 base model
        print("=" * 50)
        print("直接调用 Base Model")
        print("=" * 50)
        
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
        
        print(f"Base model outputs type: {type(base_outputs)}")
        
        if hasattr(base_outputs, 'last_hidden_state'):
            last_hidden = base_outputs.last_hidden_state
            print(f"Last hidden state shape: {last_hidden.shape}")
            print(f"Last hidden state non-zero: {(last_hidden != 0).sum().item()} / {last_hidden.numel()}")
            print(f"Last hidden state stats:")
            print(f"  Sum: {last_hidden.sum().item():.6f}")
            print(f"  Mean: {last_hidden.mean().item():.6f}")
            print(f"  Std: {last_hidden.std().item():.6f}")
            print()
        
        if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states:
            print(f"Hidden states 数量: {len(base_outputs.hidden_states)}")
            for i, hidden in enumerate(base_outputs.hidden_states[-3:]):  # 只检查最后3层
                non_zero = (hidden != 0).sum().item()
                print(f"  Layer {i}: non-zero = {non_zero} / {hidden.numel()}")
            print()
        
        # 计算 logits
        if hasattr(base_outputs, 'last_hidden_state'):
            logits = model.lm_head(base_outputs.last_hidden_state)
            print(f"Logits from base model output:")
            print(f"  Shape: {logits.shape}")
            print(f"  Non-zero: {(logits != 0).sum().item()} / {logits.numel()}")
            print(f"  Stats:")
            print(f"    Sum: {logits.sum().item():.6f}")
            print(f"    Mean: {logits.mean().item():.6f}")
            print(f"    Std: {logits.std().item():.6f}")
            print()
    
    print("=" * 50)
    print("测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



