import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """检查 embeddings"""
    
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("检查 Embeddings")
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
        # 检查 embedding
        print("=" * 50)
        print("检查 Input Embeddings")
        print("=" * 50)
        
        # 获取模型
        base_model = model.get_model()
        
        # 检查 embed_tokens
        if hasattr(base_model, 'embed_tokens'):
            embed_tokens = base_model.embed_tokens
            print(f"embed_tokens 类型: {type(embed_tokens)}")
            print(f"embed_tokens weight shape: {embed_tokens.weight.shape}")
            print(f"embed_tokens weight dtype: {embed_tokens.weight.dtype}")
            
            # 检查权重
            weight_sum = embed_tokens.weight.sum().item()
            weight_mean = embed_tokens.weight.mean().item()
            weight_std = embed_tokens.weight.std().item()
            
            print(f"embed_tokens 权重统计:")
            print(f"  Sum: {weight_sum:.6f}")
            print(f"  Mean: {weight_mean:.6f}")
            print(f"  Std: {weight_std:.6f}")
            print()
            
            # 获取输入 embedding
            input_embeds = embed_tokens(inputs.input_ids)
            print(f"Input embeddings shape: {input_embeds.shape}")
            print(f"Input embeddings dtype: {input_embeds.dtype}")
            
            # 统计信息
            embeds_sum = input_embeds.sum().item()
            embeds_mean = input_embeds.mean().item()
            embeds_std = input_embeds.std().item()
            embeds_max = input_embeds.max().item()
            embeds_min = input_embeds.min().item()
            non_zero_count = (input_embeds != 0).sum().item()
            
            print(f"Input embeddings 统计:")
            print(f"  Sum: {embeds_sum:.6f}")
            print(f"  Mean: {embeds_mean:.6f}")
            print(f"  Std: {embeds_std:.6f}")
            print(f"  Max: {embeds_max:.6f}")
            print(f"  Min: {embeds_min:.6f}")
            print(f"  非零值数量: {non_zero_count} / {input_embeds.numel()}")
            print()
            
            # 检查第一个 token 的 embedding
            first_token_embed = input_embeds[0, 0, :]
            print(f"第一个 token 的 embedding (前10个维度):")
            print(f"  {first_token_embed[:10].tolist()}")
            print()
        else:
            print("✗ 模型没有 embed_tokens")
            print()
        
        # 直接调用 forward 并检查每一步
        print("=" * 50)
        print("逐步检查 Forward 过程")
        print("=" * 50)
        
        # 手动调用 embed_tokens
        if hasattr(base_model, 'embed_tokens'):
            inputs_embeds = base_model.embed_tokens(inputs.input_ids)
            print(f"1. After embed_tokens:")
            print(f"   Shape: {inputs_embeds.shape}")
            print(f"   Non-zero: {(inputs_embeds != 0).sum().item()} / {inputs_embeds.numel()}")
            print()
            
            # 检查是否有图像处理
            print("2. 检查是否有图像 token:")
            image_token_id = model.config.image_token_id
            print(f"   Image token ID: {image_token_id}")
            n_image_tokens = (inputs.input_ids == image_token_id).sum().item()
            print(f"   Number of image tokens in input: {n_image_tokens}")
            print()
            
            # 调用模型的 forward
            print("3. 调用模型 forward:")
            outputs = model(
                input_ids=inputs.input_ids,
                inputs_embeds=inputs_embeds,  # 使用预计算的 embeddings
                output_hidden_states=True,
                return_dict=True
            )
            
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            
            print(f"   Logits shape: {logits.shape}")
            print(f"   Logits non-zero: {(logits != 0).sum().item()} / {logits.numel()}")
            
            if hidden_states:
                last_hidden = hidden_states[-1]
                print(f"   Last hidden non-zero: {(last_hidden != 0).sum().item()} / {last_hidden.numel()}")
            print()
    
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



