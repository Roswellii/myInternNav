#!/usr/bin/env python
"""
检查模型文件的完整性（MD5/SHA256 校验）
"""
import sys
import hashlib
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def calculate_hash(file_path, algorithm='sha256', chunk_size=8192):
    """
    计算文件的哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha256', 'sha1')
        chunk_size: 读取块大小
    
    Returns:
        哈希值的十六进制字符串
    """
    hash_obj = hashlib.new(algorithm)
    file_size = file_path.stat().st_size
    
    with open(file_path, 'rb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"计算 {algorithm.upper()}", leave=False) as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_obj.update(chunk)
                pbar.update(len(chunk))
    
    return hash_obj.hexdigest()


def format_file_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    """主函数"""
    model_path = Path(__file__).parent.parent / "checkpoints" / "InternVLA-N1"
    
    print("=" * 80)
    print("模型文件完整性检查 (MD5/SHA256 校验)")
    print("=" * 80)
    print(f"模型目录: {model_path}")
    print()
    
    # 查找所有 safetensors 文件
    safetensors_files = sorted(model_path.glob("model-*.safetensors"))
    
    if not safetensors_files:
        print("✗ 未找到 safetensors 文件!")
        return
    
    print(f"找到 {len(safetensors_files)} 个模型文件:")
    for f in safetensors_files:
        size = f.stat().st_size
        print(f"  - {f.name}: {format_file_size(size)}")
    print()
    
    # 计算每个文件的校验和
    results = []
    
    for file_path in safetensors_files:
        print(f"\n{'=' * 80}")
        print(f"正在处理: {file_path.name}")
        print(f"{'=' * 80}")
        
        file_size = file_path.stat().st_size
        print(f"文件大小: {format_file_size(file_size)}")
        print()
        
        # 计算 MD5
        print("计算 MD5...")
        md5_hash = calculate_hash(file_path, 'md5')
        print(f"MD5:  {md5_hash}")
        print()
        
        # 计算 SHA256
        print("计算 SHA256...")
        sha256_hash = calculate_hash(file_path, 'sha256')
        print(f"SHA256: {sha256_hash}")
        print()
        
        results.append({
            'file': file_path.name,
            'size': file_size,
            'md5': md5_hash,
            'sha256': sha256_hash,
        })
    
    # 输出汇总
    print("\n" + "=" * 80)
    print("校验和汇总")
    print("=" * 80)
    print()
    
    # 保存到文件
    output_file = model_path / "checksums.txt"
    with open(output_file, 'w') as f:
        f.write("模型文件校验和\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"文件: {result['file']}\n")
            f.write(f"大小: {format_file_size(result['size'])}\n")
            f.write(f"MD5:    {result['md5']}\n")
            f.write(f"SHA256: {result['sha256']}\n")
            f.write("-" * 80 + "\n\n")
    
    print("校验和已保存到:", output_file)
    print()
    
    # 显示表格格式
    print("文件列表:")
    print("-" * 80)
    print(f"{'文件名':<30} {'大小':<15} {'MD5':<32} {'SHA256':<64}")
    print("-" * 80)
    
    for result in results:
        md5_short = result['md5'][:16] + "..."
        sha256_short = result['sha256'][:32] + "..."
        print(f"{result['file']:<30} {format_file_size(result['size']):<15} {md5_short:<32} {sha256_short:<64}")
    
    print("-" * 80)
    print()
    
    # 特别关注 model-00004-of-00004.safetensors
    file4 = model_path / "model-00004-of-00004.safetensors"
    if file4.exists():
        print("=" * 80)
        print("特别检查: model-00004-of-00004.safetensors")
        print("=" * 80)
        print()
        
        file4_result = next((r for r in results if r['file'] == file4.name), None)
        if file4_result:
            print(f"文件大小: {format_file_size(file4_result['size'])}")
            print(f"MD5:    {file4_result['md5']}")
            print(f"SHA256: {file4_result['sha256']}")
            print()
            print("⚠️  注意: 此文件之前检查发现 99.9% 的权重全为 0")
            print("   如果这是从官方源下载的，请验证校验和是否匹配")
            print()
    
    print("=" * 80)
    print("检查完成!")
    print("=" * 80)
    print()
    print("提示:")
    print("  1. 将校验和与官方提供的校验和对比")
    print("  2. 如果校验和不匹配，文件可能损坏，需要重新下载")
    print("  3. 校验和已保存到 checksums.txt 文件")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



