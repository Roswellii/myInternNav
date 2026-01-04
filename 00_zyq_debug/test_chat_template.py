import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def main():
    """检查 chat template 是否正常"""
    
    # 配置
    model_path = str(Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("Chat Template 检查")
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
    
    # 2. 检查 chat template
    print("=" * 50)
    print("1. 检查 Chat Template 配置")
    print("=" * 50)
    
    if hasattr(processor.tokenizer, 'chat_template'):
        print(f"✓ Chat template 存在")
        if processor.tokenizer.chat_template:
            print(f"  Template 类型: {type(processor.tokenizer.chat_template)}")
            # 尝试获取模板内容
            try:
                template_str = str(processor.tokenizer.chat_template)
                print(f"  Template 长度: {len(template_str)} 字符")
                print(f"  Template 预览: {template_str[:200]}...")
            except:
                print(f"  无法获取 template 字符串")
        else:
            print("  ⚠️  Chat template 为空")
    else:
        print("  ✗ Chat template 不存在")
    
    print()
    
    # 3. 测试格式化对话
    print("=" * 50)
    print("2. 测试对话格式化")
    print("=" * 50)
    
    # 构建简单的对话
    conversation = [{
        'role': 'user',
        'content': [{'type': 'text', 'text': 'Hello, world!'}]
    }]
    
    print(f"输入对话: {conversation}")
    print()
    
    # 格式化对话
    try:
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        print("✓ 对话格式化成功")
        print()
        print("格式化后的文本:")
        print("-" * 50)
        print(repr(text))
        print("-" * 50)
        print()
        print("格式化后的文本（可读）:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        print()
    except Exception as e:
        print(f"✗ 对话格式化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 检查 tokenization
    print("=" * 50)
    print("3. 检查 Tokenization")
    print("=" * 50)
    
    try:
        inputs = processor(text=[text], return_tensors="pt")
        print(f"✓ Tokenization 成功")
        print(f"  Input IDs shape: {inputs.input_ids.shape}")
        print(f"  Input IDs (前20个): {inputs.input_ids[0][:20].tolist()}")
        print()
        
        # 检查特殊 token
        print("特殊 Token IDs:")
        if hasattr(processor.tokenizer, 'bos_token_id') and processor.tokenizer.bos_token_id:
            print(f"  BOS token ID: {processor.tokenizer.bos_token_id}")
        if hasattr(processor.tokenizer, 'eos_token_id') and processor.tokenizer.eos_token_id:
            print(f"  EOS token ID: {processor.tokenizer.eos_token_id}")
        if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id:
            print(f"  PAD token ID: {processor.tokenizer.pad_token_id}")
        print()
        
        # 解码回文本验证
        decoded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
        print("解码回文本验证:")
        print("-" * 50)
        print(repr(decoded))
        print("-" * 50)
        print()
        
    except Exception as e:
        print(f"✗ Tokenization 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 加载模型并测试生成
    print("=" * 50)
    print("4. 测试模型生成")
    print("=" * 50)
    
    print("正在加载模型...")
    try:
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        model.eval()
        print("✓ 模型加载成功")
        print()
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 处理输入
    inputs = processor(text=[text], return_tensors="pt").to(device)
    
    # 生成输出
    print("正在生成文本...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
            )
        
        print("✓ 生成成功")
        print()
        
        # 分析输出
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        
        print("生成分析:")
        print(f"  输入长度: {input_length}")
        print(f"  生成长度: {len(generated_ids)}")
        print(f"  生成的 Token IDs: {generated_ids.tolist()[:30]}")
        print()
        
        # 解码输出（不跳过特殊 token）
        output_text_full = processor.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=False
        )
        print("解码输出（包含特殊 token）:")
        print("-" * 50)
        print(repr(output_text_full))
        print("-" * 50)
        print()
        
        # 解码输出（跳过特殊 token）
        output_text = processor.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=True
        )
        print("解码输出（跳过特殊 token）:")
        print("-" * 50)
        print(repr(output_text))
        print("-" * 50)
        print()
        
        # 检查是否包含特殊字符
        if output_text.strip():
            print(f"✓ 输出不为空: {len(output_text)} 字符")
        else:
            print("⚠️  输出为空或只包含空白字符")
        
        # 检查是否全是相同字符
        if len(set(output_text.strip())) == 1:
            print(f"⚠️  输出只包含一种字符: {repr(output_text.strip()[0]) if output_text.strip() else '空'}")
        
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 50)
    print("检查完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()



