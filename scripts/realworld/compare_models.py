#!/usr/bin/env python3
"""
对比本地模型和 Hugging Face 上的 InternVLA-N1-wo-dagger 模型
"""

import json
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)


def load_local_config(model_path):
    """加载本地模型配置"""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def load_hf_config(model_name="InternRobotics/InternVLA-N1-wo-dagger"):
    """从 Hugging Face 加载模型配置"""
    try:
        from transformers import AutoConfig
        # 尝试使用 AutoConfig，如果失败则手动读取 config.json
        try:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            config_dict = config.to_dict()
        except Exception:
            # 如果 AutoConfig 失败（因为无法识别 internvla_n1），尝试直接下载 config.json
            from huggingface_hub import hf_hub_download
            import tempfile
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                cache_dir=tempfile.gettempdir()
            )
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        
        return config_dict
    except Exception as e:
        print(f"⚠️  无法从 Hugging Face 加载配置: {e}")
        print(f"   错误详情: {type(e).__name__}: {str(e)}")
        return None


def compare_configs(local_config, hf_config):
    """对比两个配置"""
    print("=" * 80)
    print("配置对比")
    print("=" * 80)
    
    if local_config is None:
        print("❌ 本地配置未找到")
        return
    
    if hf_config is None:
        print("❌ Hugging Face 配置未加载")
        print("\n本地配置:")
        print(json.dumps(local_config, indent=2, ensure_ascii=False))
        return
    
    # 单独检查 model_type
    print("\n模型类型:")
    local_model_type = local_config.get("model_type", "N/A")
    hf_model_type = hf_config.get("model_type", "N/A")
    print(f"  本地: {local_model_type}")
    print(f"  HF:   {hf_model_type}")
    if local_model_type != hf_model_type:
        print(f"  ℹ️  差异是正常的：AutoConfig 可能无法识别自定义类型 '{local_model_type}'，")
        print(f"     回退到基类 '{hf_model_type}'。这不影响模型功能。")
    
    # 核心架构参数
    core_params = [
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "hidden_act",
        "max_position_embeddings",
    ]
    
    # 特殊参数
    special_params = [
        "n_query",
        "navdp_version",
        "navdp",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ]
    
    # 注意力机制参数
    attention_params = [
        "attention_dropout",
        "rms_norm_eps",
        "rope_theta",
        "use_cache",
        "sliding_window",
        "use_sliding_window",
    ]
    
    print("\n" + "=" * 80)
    print("核心架构参数对比")
    print("=" * 80)
    print(f"{'参数':<35} {'本地模型':<25} {'HF模型':<25} {'匹配':<10}")
    print("-" * 95)
    
    all_match = True
    for param in core_params:
        local_val = local_config.get(param, "N/A")
        hf_val = hf_config.get(param, "N/A")
        match = "✓" if local_val == hf_val else "✗"
        if local_val != hf_val:
            all_match = False
        
        print(f"{param:<35} {str(local_val):<25} {str(hf_val):<25} {match:<10}")
    
    print("\n" + "=" * 80)
    print("特殊参数对比 (InternVLA-N1 特有)")
    print("=" * 80)
    print(f"{'参数':<35} {'本地模型':<25} {'HF模型':<25} {'匹配':<10}")
    print("-" * 95)
    
    for param in special_params:
        local_val = local_config.get(param, "N/A")
        hf_val = hf_config.get(param, "N/A")
        match = "✓" if local_val == hf_val else "✗"
        if local_val != hf_val:
            all_match = False
        
        # 格式化显示
        if isinstance(local_val, (list, dict)):
            local_val = str(local_val)[:50] + "..." if len(str(local_val)) > 50 else str(local_val)
        if isinstance(hf_val, (list, dict)):
            hf_val = str(hf_val)[:50] + "..." if len(str(hf_val)) > 50 else str(hf_val)
        
        print(f"{param:<35} {str(local_val):<25} {str(hf_val):<25} {match:<10}")
    
    print("\n" + "=" * 80)
    print("注意力机制参数对比")
    print("=" * 80)
    print(f"{'参数':<35} {'本地模型':<25} {'HF模型':<25} {'匹配':<10}")
    print("-" * 95)
    
    for param in attention_params:
        local_val = local_config.get(param, "N/A")
        hf_val = hf_config.get(param, "N/A")
        match = "✓" if local_val == hf_val else "✗"
        if local_val != hf_val:
            all_match = False
        
        print(f"{param:<35} {str(local_val):<25} {str(hf_val):<25} {match:<10}")
    
    # Vision config 对比
    print("\n" + "=" * 80)
    print("Vision 配置对比")
    print("=" * 80)
    local_vision = local_config.get("vision_config", {})
    hf_vision = hf_config.get("vision_config", {})
    
    vision_core_params = [
        "hidden_size",
        "out_hidden_size",
        "depth",
        "num_heads",
        "patch_size",
        "spatial_patch_size",
        "temporal_patch_size",
        "intermediate_size",
    ]
    
    vision_other_params = [
        "model_type",
        "window_size",
        "spatial_merge_size",
        "tokens_per_second",
        "in_channels",
        "hidden_act",
    ]
    
    print("\nVision 核心参数:")
    print(f"{'参数':<35} {'本地模型':<25} {'HF模型':<25} {'匹配':<10}")
    print("-" * 95)
    
    for param in vision_core_params:
        local_val = local_vision.get(param, "N/A")
        hf_val = hf_vision.get(param, "N/A")
        match = "✓" if local_val == hf_val else "✗"
        if local_val != hf_val:
            all_match = False
        
        print(f"vision.{param:<30} {str(local_val):<25} {str(hf_val):<25} {match:<10}")
    
    print("\nVision 其他参数:")
    print(f"{'参数':<35} {'本地模型':<25} {'HF模型':<25} {'匹配':<10}")
    print("-" * 95)
    
    for param in vision_other_params:
        local_val = local_vision.get(param, "N/A")
        hf_val = hf_vision.get(param, "N/A")
        match = "✓" if local_val == hf_val else "✗"
        if local_val != hf_val:
            all_match = False
        
        print(f"vision.{param:<30} {str(local_val):<25} {str(hf_val):<25} {match:<10}")
    
    # 检查 fullatt_block_indexes
    if "fullatt_block_indexes" in local_vision or "fullatt_block_indexes" in hf_vision:
        print("\nVision FullAtt Block Indexes:")
        local_blocks = local_vision.get("fullatt_block_indexes", "N/A")
        hf_blocks = hf_vision.get("fullatt_block_indexes", "N/A")
        match = "✓" if local_blocks == hf_blocks else "✗"
        if local_blocks != hf_blocks:
            all_match = False
        print(f"  vision.fullatt_block_indexes: 本地={local_blocks}, HF={hf_blocks} {match}")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ 所有关键参数匹配！本地模型和 Hugging Face 模型应该是同一版本。")
    else:
        # 检查是否只有 model_type 不同（这是正常的，因为 AutoConfig 可能无法识别自定义类型）
        only_model_type_diff = True
        for param in key_params:
            if param == "model_type":
                continue
            local_val = local_config.get(param, "N/A")
            hf_val = hf_config.get(param, "N/A")
            if local_val != hf_val:
                only_model_type_diff = False
                break
        
        if only_model_type_diff:
            print("ℹ️  仅 model_type 不同（这是正常的）")
            print("   - 本地: internvla_n1 (自定义类型)")
            print("   - HF: qwen2_5_vl (AutoConfig 回退到基类)")
            print("   ✅ 所有其他关键参数匹配！模型应该是同一版本。")
        else:
            print("⚠️  发现参数差异！建议检查模型版本。")
    print("=" * 80)


def check_model_files(local_path):
    """检查本地模型文件"""
    print("\n" + "=" * 80)
    print("本地模型文件检查")
    print("=" * 80)
    
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "generation_config.json",
    ]
    
    print("\n必需文件:")
    all_exist = True
    for file in required_files:
        file_path = os.path.join(local_path, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    # 检查模型权重文件
    print("\n模型权重文件:")
    import glob
    safetensor_files = glob.glob(os.path.join(local_path, "model-*.safetensors"))
    if safetensor_files:
        total_size = sum(os.path.getsize(f) for f in safetensor_files)
        print(f"  找到 {len(safetensor_files)} 个权重文件")
        print(f"  总大小: {total_size / 1024**3:.2f} GB")
    else:
        print("  ✗ 未找到权重文件")
        all_exist = False
    
    return all_exist


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="对比本地模型和 Hugging Face 模型")
    parser.add_argument("--local_path", type=str, default="checkpoints/InternVLA-N1",
                        help="本地模型路径")
    parser.add_argument("--hf_model", type=str, default="InternRobotics/InternVLA-N1-wo-dagger",
                        help="Hugging Face 模型名称")
    parser.add_argument("--skip_hf", action="store_true",
                        help="跳过 Hugging Face 配置加载（如果网络问题）")
    
    args = parser.parse_args()
    
    # 处理路径
    if not os.path.isabs(args.local_path):
        args.local_path = os.path.join(project_root, args.local_path)
    args.local_path = os.path.normpath(os.path.abspath(args.local_path))
    
    print("=" * 80)
    print("InternVLA-N1 模型对比工具")
    print("=" * 80)
    print(f"本地模型路径: {args.local_path}")
    print(f"Hugging Face 模型: {args.hf_model}")
    print()
    
    # 检查本地文件
    if not check_model_files(args.local_path):
        print("\n⚠️  本地模型文件不完整！")
        return
    
    # 加载配置
    print("\n" + "=" * 80)
    print("加载配置...")
    print("=" * 80)
    
    local_config = load_local_config(args.local_path)
    if local_config is None:
        print("❌ 无法加载本地配置")
        return
    
    hf_config = None
    if not args.skip_hf:
        print(f"\n从 Hugging Face 加载配置: {args.hf_model}")
        hf_config = load_hf_config(args.hf_model)
    else:
        print("\n跳过 Hugging Face 配置加载")
    
    # 对比配置
    compare_configs(local_config, hf_config)
    
    # 额外信息
    print("\n" + "=" * 80)
    print("额外信息")
    print("=" * 80)
    
    if local_config:
        print(f"\n本地模型架构: {local_config.get('architectures', ['N/A'])[0]}")
        print(f"模型类型: {local_config.get('model_type', 'N/A')}")
        print(f"Transformers 版本: {local_config.get('transformers_version', 'N/A')}")
        
        if 'generation_config.json' in os.listdir(args.local_path):
            gen_config_path = os.path.join(args.local_path, "generation_config.json")
            with open(gen_config_path, 'r') as f:
                gen_config = json.load(f)
            print(f"\n生成配置:")
            print(f"  - repetition_penalty: {gen_config.get('repetition_penalty', 'N/A')}")
            print(f"  - temperature: {gen_config.get('temperature', 'N/A')}")
            print(f"  - do_sample: {gen_config.get('do_sample', 'N/A')}")


if __name__ == "__main__":
    main()

