#!/usr/bin/env python3
"""
继续下载 InternVLA-N1-wo-dagger 模型的缺失权重文件
模型地址: https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_missing_files(model_dir):
    """检查缺失的权重文件"""
    model_dir = Path(model_dir)
    required_files = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    
    missing_files = []
    existing_files = []
    
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} 已存在 ({size_mb:.2f} MB)")
            existing_files.append(filename)
        else:
            print(f"✗ {filename} 缺失")
            missing_files.append(filename)
    
    return missing_files, existing_files

def download_missing_weights():
    """下载缺失的权重文件（使用中国镜像）"""
    model_id = "InternRobotics/InternVLA-N1-wo-dagger"
    
    # 确定保存路径
    save_dir = project_root / "checkpoints" / "InternVLA-N1-wo-dagger"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"检查模型权重文件: {save_dir}")
    print("=" * 80)
    
    # 检查缺失的文件
    missing_files, existing_files = get_missing_files(save_dir)
    
    if not missing_files:
        print("\n" + "=" * 80)
        print("✅ 所有权重文件已存在，无需下载!")
        print("=" * 80)
        return
    
    # 只有在需要下载时才导入库
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"\n需要下载 {len(missing_files)} 个文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n错误: 需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        print("或: pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)
    
    # 设置使用中国镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"\n需要下载 {len(missing_files)} 个文件:")
    for f in missing_files:
        print(f"  - {f}")
    print("\n" + "=" * 80)
    print(f"开始下载模型权重: {model_id}")
    print(f"使用镜像: https://hf-mirror.com")
    print(f"保存路径: {save_dir}")
    print("=" * 80)
    
    try:
        # 下载每个缺失的文件
        for idx, filename in enumerate(missing_files, 1):
            print(f"\n[{idx}/{len(missing_files)}] 正在下载: {filename}")
            print("-" * 80)
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=str(save_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,  # 支持断点续传
                )
                
                # 验证文件大小
                filepath = Path(downloaded_path)
                if filepath.exists():
                    size_gb = filepath.stat().st_size / (1024 ** 3)
                    print(f"✓ {filename} 下载完成 ({size_gb:.2f} GB)")
                else:
                    print(f"✗ {filename} 下载失败: 文件不存在")
                    
            except Exception as e:
                print(f"✗ {filename} 下载失败: {e}")
                print("继续下载下一个文件...")
                continue
        
        # 再次检查所有文件
        print("\n" + "=" * 80)
        print("下载完成，验证文件完整性...")
        print("=" * 80)
        missing_files_after, existing_files_after = get_missing_files(save_dir)
        
        if not missing_files_after:
            print("\n" + "=" * 80)
            print("✅ 所有权重文件下载完成!")
            print(f"模型保存在: {save_dir}")
            print("=" * 80)
        else:
            print(f"\n⚠️  仍有 {len(missing_files_after)} 个文件缺失:")
            for f in missing_files_after:
                print(f"  - {f}")
            print("\n请重新运行此脚本继续下载")
        
    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 如果模型需要认证，请先登录:")
        print("   huggingface-cli login")
        print("3. 检查磁盘空间是否充足")
        print("4. 重新运行此脚本继续下载")
        sys.exit(1)

if __name__ == "__main__":
    download_missing_weights()

