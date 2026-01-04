import sys
import os
import copy
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """详细调试生成过程"""
    
    # 配置
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("详细调试生成过程")
    print("=" * 50)
    
    # 1. 加载 Processor
    print("正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    print("✓ Processor 加载成功")
    print()
    
    # 2. 加载模型
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
    
    # 3. 构建对话
    conversation = [{
        'role': 'user',
        'content': [{'type': 'text', 'text': 'Hello, world!'}]
    }]
    
    # 格式化对话
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("格式化后的文本:")
    print(repr(text))
    print()
    
    # 处理输入
    inputs = processor(text=[text], return_tensors="pt").to(device)
    
    print("输入信息:")
    print(f"  Input IDs shape: {inputs.input_ids.shape}")
    print(f"  Input IDs: {inputs.input_ids[0].tolist()[:30]}...")
    print()
    
    # 4. 先测试单步 forward，看看 logits
    print("=" * 50)
    print("测试单步 Forward")
    print("=" * 50)
    
    with torch.no_grad():
        # 单步 forward
        outputs_forward = model(**inputs)
        logits = outputs_forward.logits
        
        print(f"Logits shape: {logits.shape}")
        print(f"Vocab size: {model.config.vocab_size}")
        print()
        
        # 获取最后一个位置的 logits
        last_logits = logits[0, -1, :]  # [vocab_size]
        
        # 获取 top-k tokens
        top_k = 10
        top_values, top_indices = torch.topk(last_logits, top_k)
        
        print(f"Top {top_k} tokens (单步 forward):")
        for i, (val, idx) in enumerate(zip(top_values, top_indices)):
            token_str = processor.tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token ID {idx.item():6d}: {repr(token_str):30s} (logit: {val.item():.2f})")
        print()
        
        # 检查 token 0 的 logit
        token_0_logit = last_logits[0].item()
        print(f"Token 0 (!) logit: {token_0_logit:.2f}")
        print()
    
    # 5. 测试生成（只生成几个 token）
    print("=" * 50)
    print("测试生成（max_new_tokens=5）")
    print("=" * 50)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # 只生成5个token便于调试
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            raw_input_ids=copy.deepcopy(inputs.input_ids),
            eos_token_id=[151645, 151643],
            output_scores=True,  # 输出每个步骤的分数
        )
    
    output_ids = outputs.sequences
    input_length = inputs.input_ids.shape[1]
    generated_ids = output_ids[0][input_length:]
    
    print(f"生成的 Token IDs: {generated_ids.tolist()}")
    print()
    
    # 解码每个 token
    print("生成的 Token 详情:")
    for i, token_id in enumerate(generated_ids.tolist()):
        token_str = processor.tokenizer.decode([token_id])
        print(f"  Step {i+1}: Token ID {token_id:6d} = {repr(token_str)}")
    print()
    
    # 如果有 scores，显示每个步骤的 top tokens
    if hasattr(outputs, 'scores') and outputs.scores:
        print("每个生成步骤的 Top 5 tokens:")
        for step, scores in enumerate(outputs.scores):
            top_values, top_indices = torch.topk(scores[0], 5)
            print(f"  Step {step+1}:")
            for val, idx in zip(top_values, top_indices):
                token_str = processor.tokenizer.decode([idx.item()])
                print(f"    Token {idx.item():6d}: {repr(token_str):30s} (logit: {val.item():.2f})")
        print()
    
    # 6. 检查模型配置
    print("=" * 50)
    print("模型配置检查")
    print("=" * 50)
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"EOS token ID: {model.config.eos_token_id}")
    print(f"BOS token ID: {model.config.bos_token_id}")
    print(f"PAD token ID: {model.config.pad_token_id}")
    print()
    
    # 7. 检查 generation config
    print("=" * 50)
    print("Generation Config")
    print("=" * 50)
    if hasattr(model, 'generation_config'):
        gen_config = model.generation_config
        print(f"do_sample: {gen_config.do_sample}")
        print(f"temperature: {gen_config.temperature}")
        print(f"top_k: {gen_config.top_k}")
        print(f"top_p: {gen_config.top_p}")
        print(f"eos_token_id: {gen_config.eos_token_id}")
    print()
    
    print("=" * 50)
    print("调试完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



