# Habitat EGL OpenGL 错误最终解决方案

## 问题
```
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
```

## 根本原因
当前安装的 `habitat-sim 0.3.3` 在初始化 OpenGL 上下文时失败。这可能是由于：
1. habitat-sim 编译时 EGL 支持配置不正确
2. 与当前 NVIDIA 驱动版本不兼容
3. conda 环境中的 EGL 库与系统库冲突

## 已尝试的修复方法（均失败）
1. ✅ 设置 EGL 环境变量
2. ✅ 使用 LD_PRELOAD 强制使用 NVIDIA 库
3. ✅ 设置 LD_LIBRARY_PATH
4. ✅ 尝试禁用渲染器（但 habitat-sim 仍在初始化时尝试创建 OpenGL 上下文）
5. ✅ 尝试 headless 模式（同样失败）

## 推荐解决方案

### 方案1: 重新安装 habitat-sim（推荐）
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
conda uninstall habitat-sim -y
conda install habitat-sim -c conda-forge -c aihabitat --force-reinstall
```

### 方案2: 使用 Docker（最可靠）
项目中有 Docker 配置，Docker 环境可能已经正确配置了 EGL：
```bash
# 查看项目中的 Docker 运行脚本
cat 00_zyq_debug/readme.md
```

### 方案3: 从源码编译 habitat-sim
如果 conda 版本有问题，可以从源码编译支持 EGL 的版本。

## 当前代码修改
已完成的代码修改（即使问题解决后也保留）：
- `habitat_vln_evaluator.py`: 添加了 EGL 环境变量配置
- `habitat_env.py`: 添加了 EGL 环境变量配置和错误处理
- `vln_r2r.yaml`: 设置 `gpu_device_id: 0`

这些修改在问题解决后仍然有用。

## 下一步
建议先尝试方案1（重新安装），如果不行则使用方案2（Docker）。













