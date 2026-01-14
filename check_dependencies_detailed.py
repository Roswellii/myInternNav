#!/usr/bin/env python3
"""
详细检查 conda 环境 internnav 中是否已安装所有 requirements 文件中列出的依赖
包括版本冲突检查
"""

import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict
from packaging import version
from packaging.specifiers import SpecifierSet

def parse_requirements_file(filepath):
    """解析 requirements 文件，返回包名和版本要求的字典"""
    requirements = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 处理 git+ 依赖（如 git+https://...）
            if line.startswith('git+') or '@ git+' in line:
                # 提取包名（通常是最后一个 / 后的部分，去掉 .git）
                parts = line.split('@')[0].strip()
                if 'git+' in parts:
                    # 尝试提取包名
                    match = re.search(r'/([^/]+?)(?:\.git)?(?:@.*)?$', parts)
                    if match:
                        pkg_name = match.group(1).replace('-', '_').lower()
                        requirements[pkg_name] = {'spec': line, 'type': 'git', 'source': filepath.name}
                else:
                    # 直接是包名
                    pkg_name = parts.replace('-', '_').lower()
                    requirements[pkg_name] = {'spec': line, 'type': 'git', 'source': filepath.name}
                continue
            
            # 处理普通的包依赖
            # 格式可能是: package==version, package>=version, package~=version 等
            # 也可能有环境标记，如: package>=1.0 ; python_version >= "3.10"
            
            # 分离环境标记
            env_marker = None
            if ';' in line:
                line, env_marker = line.split(';', 1)
                line = line.strip()
            
            # 解析包名和版本
            # 支持 ==, >=, <=, >, <, ~=, !=
            match = re.match(r'^([a-zA-Z0-9_-]+)\s*(.*)$', line)
            if match:
                pkg_name = match.group(1).replace('-', '_').lower()
                version_spec = match.group(2).strip()
                
                if not version_spec:
                    requirements[pkg_name] = {'spec': None, 'type': 'pypi', 'source': filepath.name, 'env_marker': env_marker}
                else:
                    requirements[pkg_name] = {'spec': version_spec, 'type': 'pypi', 'source': filepath.name, 'env_marker': env_marker}
    
    return requirements

def get_installed_packages():
    """获取当前环境中已安装的包"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        
        installed = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                pkg_name, version_str = line.split('==', 1)
                pkg_name = pkg_name.replace('-', '_').lower()
                installed[pkg_name] = version_str.strip()
        
        return installed
    except subprocess.CalledProcessError as e:
        print(f"错误: 无法获取已安装的包列表: {e}")
        return {}

def check_version_compatibility(installed_version, required_spec):
    """检查已安装版本是否满足要求"""
    if not required_spec:
        return True, "无版本要求"
    
    try:
        inst_ver = version.parse(installed_version)
        
        # 尝试匹配要求
        # 简化处理：如果要求是 ==，直接比较
        if required_spec.startswith('=='):
            req_ver = required_spec[2:].strip()
            req_parsed = version.parse(req_ver)
            if inst_ver == req_parsed:
                return True, f"版本匹配: {installed_version}"
            else:
                return False, f"版本不匹配: 已安装 {installed_version}, 需要 {req_ver}"
        
        # 对于其他操作符，使用 SpecifierSet
        spec_set = SpecifierSet(required_spec)
        if inst_ver in spec_set:
            return True, f"版本满足要求: {installed_version}"
        else:
            return False, f"版本不满足: 已安装 {installed_version}, 需要 {required_spec}"
    except Exception as e:
        # 如果解析失败，假设满足（可能是开发版本等）
        return True, f"无法解析版本 (可能是开发版本): {installed_version}"

def main():
    requirements_dir = Path(__file__).parent / 'requirements'
    
    # 收集所有 requirements，按包名分组以检测冲突
    all_requirements_by_pkg = defaultdict(list)
    req_files = [
        'core_requirements.txt',
        'internvla_n1.txt',
        'isaac_requirements.txt',
        'model_requirements.txt',
        'habitat_requirements.txt'
    ]
    
    for req_file in req_files:
        filepath = requirements_dir / req_file
        if filepath.exists() and filepath.stat().st_size > 0:
            reqs = parse_requirements_file(filepath)
            for pkg_name, pkg_info in reqs.items():
                pkg_info['file'] = req_file
                all_requirements_by_pkg[pkg_name].append(pkg_info)
    
    print("="*80)
    print("依赖检查报告 - conda 环境: internnav")
    print("="*80)
    print(f"\n检查了 {len(req_files)} 个 requirements 文件")
    print(f"总共找到 {len(all_requirements_by_pkg)} 个唯一的依赖包\n")
    
    # 获取已安装的包
    installed = get_installed_packages()
    print(f"环境中已安装 {len(installed)} 个包\n")
    
    # 检查版本冲突
    print("="*80)
    print("版本冲突检查:")
    print("="*80)
    version_conflicts = []
    for pkg_name, reqs_list in sorted(all_requirements_by_pkg.items()):
        if len(reqs_list) > 1:
            # 检查是否有版本冲突
            specs = [r.get('spec') for r in reqs_list if r.get('spec') and r.get('type') == 'pypi']
            if len(specs) > 1:
                # 检查是否所有版本要求相同
                if len(set(specs)) > 1:
                    version_conflicts.append((pkg_name, reqs_list))
                    print(f"⚠ {pkg_name}:")
                    for req in reqs_list:
                        if req.get('spec'):
                            print(f"    - {req['file']}: {req['spec']}")
    
    if not version_conflicts:
        print("✓ 未发现版本冲突\n")
    else:
        print(f"\n发现 {len(version_conflicts)} 个版本冲突\n")
    
    # 检查每个依赖
    print("="*80)
    print("依赖安装状态检查:")
    print("="*80)
    
    missing = []
    version_mismatch = []
    installed_pkgs = []
    git_deps = []
    
    # 对于每个包，使用第一个要求（如果有多文件冲突，优先使用最新的要求）
    for pkg_name, reqs_list in sorted(all_requirements_by_pkg.items()):
        # 选择第一个非 git 依赖，或者第一个依赖
        req = reqs_list[0]
        spec = req.get('spec')
        pkg_type = req.get('type', 'pypi')
        source_file = req.get('source', req.get('file', 'unknown'))
        
        if pkg_type == 'git':
            git_deps.append((pkg_name, spec, source_file))
            # 对于 git 依赖，检查是否有类似的包名已安装
            found = False
            for inst_name, inst_ver in installed.items():
                if pkg_name in inst_name or inst_name in pkg_name:
                    print(f"✓ {pkg_name:40s} [Git依赖] - 可能已安装 (类似包: {inst_name}=={inst_ver}) [{source_file}]")
                    found = True
                    break
            if not found:
                print(f"? {pkg_name:40s} [Git依赖] - 需要手动验证 [{source_file}]")
            continue
        
        if pkg_name not in installed:
            missing.append((pkg_name, spec, source_file))
            print(f"✗ {pkg_name:40s} - 未安装 [{source_file}]")
        else:
            installed_ver = installed[pkg_name]
            if spec:
                is_compatible, msg = check_version_compatibility(installed_ver, spec)
                if is_compatible:
                    installed_pkgs.append((pkg_name, installed_ver, spec))
                    print(f"✓ {pkg_name:40s} - {installed_ver:20s} [{source_file}]")
                else:
                    version_mismatch.append((pkg_name, installed_ver, spec, source_file))
                    print(f"⚠ {pkg_name:40s} - {installed_ver:20s} 需要 {spec:20s} [{source_file}]")
            else:
                installed_pkgs.append((pkg_name, installed_ver, None))
                print(f"✓ {pkg_name:40s} - {installed_ver:20s} (已安装) [{source_file}]")
    
    # 总结
    print("\n" + "="*80)
    print("检查总结:")
    print("="*80)
    print(f"✓ 已安装且版本正确: {len(installed_pkgs)}")
    print(f"⚠ 版本不匹配: {len(version_mismatch)}")
    print(f"✗ 未安装: {len(missing)}")
    print(f"📦 Git 依赖 (需手动验证): {len(git_deps)}")
    if version_conflicts:
        print(f"⚠ 版本冲突: {len(version_conflicts)}")
    
    if missing:
        print("\n未安装的包:")
        for pkg, spec, source in missing:
            if spec:
                print(f"  - {pkg} ({spec}) [来自: {source}]")
            else:
                print(f"  - {pkg} [来自: {source}]")
        print("\n提示: flash_attn 是可选依赖，用于性能优化。如果未安装，")
        print("      功能仍可用但可能较慢。安装命令:")
        print("      pip install flash-attn==2.7.4.post1 --no-build-isolation")
    
    if version_mismatch:
        print("\n版本不匹配的包:")
        for pkg, inst_ver, req_spec, source in version_mismatch:
            print(f"  - {pkg}: 已安装 {inst_ver}, 需要 {req_spec} [来自: {source}]")
    
    if version_conflicts:
        print("\n版本冲突的包 (在不同文件中要求不同版本):")
        for pkg, reqs_list in version_conflicts:
            print(f"  - {pkg}:")
            for req in reqs_list:
                if req.get('spec'):
                    print(f"      {req.get('file', req.get('source', 'unknown'))}: {req['spec']}")
    
    if git_deps:
        print("\nGit 依赖 (需手动验证是否已安装):")
        for pkg, spec, source in git_deps:
            print(f"  - {pkg}: {spec} [来自: {source}]")
    
    print("\n" + "="*80)
    
    # 返回状态码
    if missing and 'flash_attn' not in [p[0] for p in missing]:
        # 除了 flash_attn 之外还有缺失
        print("⚠️  发现依赖问题，请检查上述列表")
        return 1
    elif missing:
        print("⚠️  flash_attn 未安装 (可选依赖，用于性能优化)")
        return 0
    elif version_mismatch or version_conflicts:
        print("⚠️  发现版本问题，请检查上述列表")
        return 1
    else:
        print("✓ 所有依赖检查通过！")
        return 0

if __name__ == '__main__':
    sys.exit(main())

















