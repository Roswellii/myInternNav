#!/usr/bin/env python3
"""
简单测试脚本：测试 InternVLA-N1 模型的文本生成能力
用于诊断模型是否会出现重复生成感叹号的问题
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

# 添加项目路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# 延迟导入以避免循环导入问题
def _get_project_root():
    """获取项目根路径"""
    return project_root

def _load_model_with_config(model_path, device="cuda:0", local_files_only=True):
    """
    从模型路径加载配置，然后使用 Qwen2_5_VLForConditionalGeneration 加载模型
    这样可以确保使用正确的配置，避免配置不匹配问题
    """
    print("ℹ️  从模型路径加载配置...")
    
    # 手动读取 config.json 并创建配置对象
    # 因为 AutoConfig 无法识别 internvla_n1 类型
    import json
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # 将 model_type 改为 qwen2_5_vl，以便使用 Qwen2_5_VLConfig
    original_model_type = config_dict.get('model_type', 'qwen2_5_vl')
    config_dict['model_type'] = 'qwen2_5_vl'
    
    # 创建配置对象
    config = Qwen2_5_VLConfig.from_dict(config_dict)
    
    print(f"✓ 配置加载完成（从 config.json 手动加载）")
    print(f"  - 原始 model_type: {original_model_type}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - hidden_size: {config.hidden_size}")
    
    # 使用配置加载模型
    print("ℹ️  使用 Qwen2_5_VLForConditionalGeneration 加载模型（避免循环导入）")
    print("   注意：这是基类，对于文本生成测试应该足够")
    
    return Qwen2_5_VLForConditionalGeneration, config


def load_model_and_processor(model_path, device="cuda:0"):
    """加载模型和processor"""
    print(f"=" * 80)
    print(f"加载模型和processor")
    print(f"=" * 80)
    
    # 获取项目根路径
    PROJECT_ROOT_PATH = _get_project_root()
    
    # 处理模型路径
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT_PATH, model_path)
    model_path = os.path.normpath(os.path.abspath(model_path))
    
    # 检查本地文件
    config_file = os.path.join(model_path, "config.json")
    local_files_only = os.path.exists(config_file)
    
    if local_files_only:
        print(f"✓ 从本地路径加载模型: {model_path}")
    else:
        print(f"⚠️  本地模型文件未找到，将从Hugging Face Hub下载")
        print(f"   查找路径: {config_file}")
    
    # 获取模型类和配置
    # 由于循环导入问题，使用 Qwen2_5_VLForConditionalGeneration
    # 这是 InternVLAN1ForCausalLM 的基类，对于文本生成测试应该足够
    ModelClass, config = _load_model_with_config(model_path, device, local_files_only)
    
    # 加载模型（使用从模型路径加载的配置）
    print(f"\n[1/2] 加载模型...")
    model = ModelClass.from_pretrained(
        model_path,
        config=config,  # 使用加载的配置
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
        local_files_only=local_files_only,
        trust_remote_code=True,  # 可能需要信任远程代码
    )
    model.eval()
    print(f"✓ 模型加载完成，设备: {device}")
    
    # 加载processor
    print(f"\n[2/2] 加载processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    processor.tokenizer.padding_side = 'left'
    print(f"✓ Processor加载完成")
    
    return model, processor


def test_text_only_generation(model, processor, device, test_cases, use_repetition_penalty=True):
    """测试纯文本生成（无图像）"""
    print(f"\n" + "=" * 80)
    print(f"测试纯文本生成（无图像输入）")
    print(f"=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"测试用例 {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"输入文本: {test_case['text']}")
        
        # 构建对话历史
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": test_case['text']}
                ]
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"\nChat template后的文本 (前200字符):")
        print(f"{text[:200]}...")
        
        # 处理输入（无图像）
        inputs = processor(text=[text], return_tensors="pt").to(device)
        print(f"\n输入信息:")
        print(f"  - input_ids shape: {inputs.input_ids.shape}")
        print(f"  - input_ids长度: {inputs.input_ids.shape[1]}")
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": test_case.get('max_new_tokens', 128),
            "do_sample": False,
            "use_cache": True,
        }
        
        if use_repetition_penalty:
            generation_kwargs["repetition_penalty"] = 1.1
            print(f"\n生成参数: {generation_kwargs}")
        else:
            print(f"\n生成参数: {generation_kwargs} (无repetition_penalty)")
        
        # 生成
        print(f"\n>>> 开始生成...")
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
                return_dict_in_generate=True,
            )
        t1 = time.time()
        
        output_ids = outputs.sequences
        generated_text = processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"<<< 生成完成，耗时: {t1 - t0:.4f}s")
        print(f"\n生成结果:")
        print(f"  - output_ids shape: {output_ids.shape}")
        print(f"  - 生成token数量: {output_ids.shape[1] - inputs.input_ids.shape[1]}")
        print(f"  - 生成文本长度: {len(generated_text)}")
        print(f"  - 生成文本内容:")
        print(f"    {repr(generated_text)}")
        print(f"\n生成文本（可读格式）:")
        print(f"    {generated_text}")
        
        # 分析输出
        print(f"\n输出分析:")
        unique_chars = set(generated_text)
        print(f"  - 唯一字符数: {len(unique_chars)}")
        print(f"  - 唯一字符: {sorted(unique_chars)}")
        
        # 检查是否全是重复字符
        if len(unique_chars) == 1:
            char = list(unique_chars)[0]
            count = len(generated_text)
            print(f"  ⚠️  警告: 输出全是重复字符 '{char}' (共{count}个)")
        elif len(unique_chars) <= 3:
            print(f"  ⚠️  警告: 输出字符种类很少，可能存在重复生成问题")
        
        # 检查是否包含感叹号
        exclamation_count = generated_text.count('!')
        if exclamation_count > len(generated_text) * 0.5:
            print(f"  ⚠️  警告: 输出包含大量感叹号 ({exclamation_count}/{len(generated_text)})")
        
        # 检查是否包含数字
        has_digits = any(c.isdigit() for c in generated_text)
        print(f"  - 包含数字: {has_digits}")
        
        # 检查是否包含动作符号
        action_symbols = ['STOP', '↑', '←', '→', '↓']
        found_actions = [sym for sym in action_symbols if sym in generated_text]
        print(f"  - 包含动作符号: {found_actions if found_actions else '无'}")


def test_with_simple_image(model, processor, device):
    """测试带简单图像的生成"""
    print(f"\n" + "=" * 80)
    print(f"测试带图像的生成")
    print(f"=" * 80)
    
    # 创建一个简单的测试图像（全黑）
    from PIL import Image
    import numpy as np
    
    test_image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
    print(f"创建测试图像: {test_image.size}, mode={test_image.mode}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "What do you see in this image? Please describe it briefly."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(text=[text], images=[test_image], return_tensors="pt").to(device)
    print(f"\n输入信息:")
    print(f"  - input_ids shape: {inputs.input_ids.shape}")
    print(f"  - pixel_values shape: {inputs.pixel_values.shape if hasattr(inputs, 'pixel_values') else 'N/A'}")
    
    print(f"\n>>> 开始生成...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.1,
            return_dict_in_generate=True,
        )
    t1 = time.time()
    
    output_ids = outputs.sequences
    generated_text = processor.tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    print(f"<<< 生成完成，耗时: {t1 - t0:.4f}s")
    print(f"\n生成结果:")
    print(f"    {repr(generated_text)}")


def main():
    parser = argparse.ArgumentParser(description="测试InternVLA-N1模型的文本生成能力")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-wo-dagger",
                        help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备 (cuda:0, cpu等)")
    parser.add_argument("--no_repetition_penalty", action="store_true",
                        help="不使用repetition_penalty（用于对比测试）")
    parser.add_argument("--test_image", action="store_true",
                        help="也测试带图像的生成")
    parser.add_argument("--use_base_model", action="store_true",
                        help="使用 Qwen2_5_VLForConditionalGeneration 基类（避免循环导入，默认行为）")
    
    args = parser.parse_args()
    
    print(f"=" * 80)
    print(f"InternVLA-N1 模型文本生成测试")
    print(f"=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"使用repetition_penalty: {not args.no_repetition_penalty}")
    
    # 加载模型
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    # 定义测试用例
    test_cases = [
        {
            "name": "简单问候",
            "text": "Hello, how are you?",
            "max_new_tokens": 50,
        },
        {
            "name": "导航指令（无图像）",
            "text": "You are an autonomous navigation assistant. Your task is to move forward and stop in front of the red backpack. Where should you go next?",
            "max_new_tokens": 128,
        },
        {
            "name": "简单问答",
            "text": "What is 2+2?",
            "max_new_tokens": 30,
        },
        {
            "name": "动作指令",
            "text": "Please output STOP when you have successfully completed the task.",
            "max_new_tokens": 50,
        },
    ]
    
    # 测试纯文本生成
    use_repetition_penalty = not args.no_repetition_penalty
    test_text_only_generation(
        model, processor, args.device, test_cases, 
        use_repetition_penalty=use_repetition_penalty
    )
    
    # 测试带图像的生成
    if args.test_image:
        test_with_simple_image(model, processor, args.device)
    
    print(f"\n" + "=" * 80)
    print(f"测试完成")
    print(f"=" * 80)


if __name__ == "__main__":
    main()

