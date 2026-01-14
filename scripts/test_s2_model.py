#!/usr/bin/env python3
"""
测试脚本：单独测试 internvla_n1_agent_realworld.py 中的 s2 模型
从指定文件夹读取图像和深度数据，调用 step_s2 方法进行测试
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# 添加 diffusion-policy 路径
sys.path.insert(0, str(project_root / "src" / "diffusion-policy"))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


def load_test_data(data_dir, step_idx=1):
    """
    从数据文件夹加载测试数据
    
    Args:
        data_dir: 数据文件夹路径
        step_idx: 步骤索引（例如 1 表示 step_0001）
    
    Returns:
        rgb: RGB图像 (numpy array, uint8, HxWx3)
        depth: 深度图像 (numpy array, float32, HxW)
    """
    data_dir = Path(data_dir)
    
    # 加载RGB图像
    rgb_path = data_dir / f"step_{step_idx:04d}_image.jpg"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB图像不存在: {rgb_path}")
    
    rgb_image = Image.open(rgb_path)
    rgb_image = rgb_image.convert('RGB')
    rgb = np.asarray(rgb_image)
    print(f"加载RGB图像: {rgb_path}, 形状: {rgb.shape}, 数据类型: {rgb.dtype}")
    
    # 加载深度数据
    depth_path = data_dir / f"step_{step_idx:04d}_depth.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"深度数据不存在: {depth_path}")
    
    depth = np.load(depth_path)
    print(f"加载深度数据: {depth_path}, 形状: {depth.shape}, 数据类型: {depth.dtype}")
    
    # 确保深度数据是float32格式（如果是其他格式，可能需要转换）
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
        print(f"转换深度数据类型为 float32")
    
    return rgb, depth


def main():
    parser = argparse.ArgumentParser(description="测试 s2 模型")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cuda:0 或 cpu)")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1", help="模型路径")
    parser.add_argument("--resize_w", type=int, default=384, help="图像宽度")
    parser.add_argument("--resize_h", type=int, default=384, help="图像高度")
    parser.add_argument("--num_history", type=int, default=8, help="历史帧数")
    parser.add_argument("--plan_step_gap", type=int, default=8, help="计划步长间隔")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scripts/realworld/output/runs12-31-1120",
        help="测试数据文件夹路径",
    )
    parser.add_argument("--step_idx", type=int, default=1, help="要测试的步骤索引 (例如: 1 表示 step_0001)")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin.",
        help="导航指令",
    )
    
    args = parser.parse_args()
    
    # 设置相机内参（默认值，可以根据实际情况修改）
    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    
    print("=" * 80)
    print("初始化 InternVLAN1AsyncAgent...")
    print(f"设备: {args.device}")
    print(f"模型路径: {args.model_path}")
    print("=" * 80)
    
    # 初始化agent
    agent = InternVLAN1AsyncAgent(args)
    
    print("\n" + "=" * 80)
    print("加载测试数据...")
    print("=" * 80)
    
    # 加载测试数据
    rgb, depth = load_test_data(args.data_dir, args.step_idx)
    
    # 准备pose（单位矩阵，表示初始位置）
    pose = np.eye(4)
    
    print("\n" + "=" * 80)
    print("调用 step_s2 方法进行测试...")
    print(f"指令: {args.instruction}")
    print("=" * 80)
    
    # 调用 step_s2 方法
    output_action, output_latent, output_pixel = agent.step_s2(
        rgb=rgb,
        depth=depth,
        pose=pose,
        instruction=args.instruction,
        intrinsic=args.camera_intrinsic,
        look_down=False,
    )
    
    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)
    print(f"output_action: {output_action}")
    print(f"output_latent: {output_latent}")
    if output_latent is not None:
        print(f"  - 形状: {output_latent.shape if hasattr(output_latent, 'shape') else 'N/A'}")
        print(f"  - 类型: {type(output_latent)}")
    print(f"output_pixel: {output_pixel}")
    print(f"LLM输出: {agent.llm_output}")
    print("=" * 80)
    
    # 保存结果信息
    result_file = Path(agent.save_dir) / "test_s2_result.txt"
    with open(result_file, "w") as f:
        f.write("S2模型测试结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试数据: {args.data_dir}/step_{args.step_idx:04d}\n")
        f.write(f"指令: {args.instruction}\n")
        f.write(f"output_action: {output_action}\n")
        f.write(f"output_latent: {output_latent}\n")
        f.write(f"output_pixel: {output_pixel}\n")
        f.write(f"LLM输出: {agent.llm_output}\n")
    
    print(f"\n结果已保存到: {result_file}")
    print(f"调试图像保存在: {agent.save_dir}")


if __name__ == "__main__":
    main()

