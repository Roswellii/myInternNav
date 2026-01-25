# Habitat EGL渲染问题修复总结

## 问题描述
运行habitat评估时出现错误：
```
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
```

## 已完成的修改

### 1. 代码修改
- **`internnav/habitat_extensions/habitat_vln_evaluator.py`**: 添加了EGL环境变量设置
- **`internnav/habitat_extensions/habitat_env.py`**: 添加了EGL环境变量设置
- **`scripts/eval/configs/vln_r2r.yaml`**: 设置 `gpu_device_id: 0`

### 2. 环境变量设置
代码中已自动设置以下环境变量：
- `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d`
- `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- `EGL_VISIBLE_DEVICES=0`
- `CUDA_VISIBLE_DEVICES=0`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:...`

### 3. 运行脚本
- **`run_habitat_gpu.sh`**: 使用GPU渲染的脚本（已配置所有环境变量）

## 当前状态
问题仍然存在：habitat-sim在初始化OpenGL上下文时失败。

## 可能的解决方案

### 方案1: 安装Xvfb虚拟显示（推荐尝试）
```bash
sudo apt-get install xvfb
./run_habitat_with_xvfb.sh
```

### 方案2: 重新安装habitat-sim（支持EGL的版本）
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
conda uninstall habitat-sim -y
conda install habitat-sim -c conda-forge -c aihabitat --force-reinstall
```

### 方案3: 检查habitat-sim编译选项
当前安装的habitat-sim可能不支持EGL渲染。可能需要：
- 从源码编译habitat-sim，确保启用EGL支持
- 或使用预编译的支持EGL的版本

### 方案4: 使用Docker（如果本地环境有问题）
项目中有Docker配置，可以在Docker容器中运行，容器环境可能已经正确配置了EGL。

## 测试命令
```bash
# 使用修改后的脚本
./run_habitat_gpu.sh

# 或手动运行
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
```

## 下一步
1. 尝试安装Xvfb并运行 `./run_habitat_with_xvfb.sh`
2. 如果仍然失败，考虑重新安装habitat-sim
3. 或者使用Docker环境运行












