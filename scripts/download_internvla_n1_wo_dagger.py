#!/usr/bin/env python3
"""
下载 InternVLA-N1-wo-dagger 模型
模型地址: https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_model():
    """下载 InternVLA-N1-wo-dagger 模型（使用中国镜像）"""
    try:
        from huggingface_hub import snapshot_download
        import os
    except ImportError:
        print("错误: 需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        print("或: pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)
    
    model_id = "InternRobotics/InternVLA-N1-wo-dagger"
    
    # 设置使用中国镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 确定保存路径
    save_dir = project_root / "checkpoints" / "InternVLA-N1-wo-dagger"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"使用镜像: https://hf-mirror.com")
    print(f"保存路径: {save_dir}")
    print("-" * 80)
    
    try:
        # 下载模型（使用镜像，通过环境变量自动使用）
        snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            resume_download=True,  # 支持断点续传
        )
        print("-" * 80)
        print(f"✅ 模型下载完成!")
        print(f"模型保存在: {save_dir}")
        print(f"\n使用方法:")
        print(f"  在代码中设置 model_path 为: checkpoints/InternVLA-N1-wo-dagger")
        print(f"  或绝对路径: {save_dir}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 如果模型需要认证，请先登录:")
        print("   huggingface-cli login")
        print("3. 检查磁盘空间是否充足")
        sys.exit(1)

if __name__ == "__main__":
    download_model()

