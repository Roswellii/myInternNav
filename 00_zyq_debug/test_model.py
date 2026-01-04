import sys
import os
import copy
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """最简单的 Hello World 测试 - 加载 InternVLA-N1 模型并生成文本"""
    
    # 配置
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("Hello World 测试 - InternVLA-N1 模型")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print()
    
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
    
    # 3. 简单的 Hello World 测试
    print("=" * 50)
    print("开始 Hello World 测试")
    print("=" * 50)
    
    # 构建简单的对话
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
    
    # 处理输入
    inputs = processor(text=[text], return_tensors="pt").to(device)
    
    # 生成输出（参考原始代码的参数）
    print("正在生成文本...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            raw_input_ids=copy.deepcopy(inputs.input_ids),  # 重要：原始代码中使用了这个参数
            eos_token_id=[151645, 151643],  # 从 generation_config.json 中获取
        )
    
    # 解码输出（参考原始代码的方式）
    output_ids = outputs.sequences
    input_length = inputs.input_ids.shape[1]
    generated_ids = output_ids[0][input_length:]
    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print()
    print("=" * 50)
    print("测试结果:")
    print("=" * 50)
    print(f"输入: Hello, world!")
    print(f"输出: {output_text}")
    print("=" * 50)
    print("✓ Hello World 测试完成!")


if __name__ == "__main__":
    main()
