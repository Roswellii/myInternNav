# CUDA 12.1 升级指南

## 当前状态

- **当前 CUDA 版本**: 11.8
- **当前 PyTorch**: 2.5.1+cu118 (CUDA 11.8)
- **NVIDIA 驱动**: 580.95.05 (支持 CUDA 13.0) ✅

## 升级步骤

### 方法 1: 使用脚本自动升级（推荐）

#### 步骤 1: 升级 CUDA 到 12.1

```bash
# 需要 root 权限
sudo bash scripts/realworld/upgrade_cuda_12.1.sh
```

脚本会：
1. 检查当前 CUDA 版本
2. 检查 NVIDIA 驱动兼容性
3. 下载并安装 CUDA 12.1
4. 设置环境变量
5. 创建符号链接

#### 步骤 2: 重新加载环境变量

```bash
source /etc/profile.d/cuda.sh
# 或重新登录终端
```

#### 步骤 3: 验证 CUDA 安装

```bash
nvcc --version
# 应该显示 CUDA 12.1
```

#### 步骤 4: 升级 PyTorch 以支持 CUDA 12.1

```bash
# 激活 conda 环境（如果使用）
conda activate internnav

# 运行升级脚本
bash scripts/realworld/upgrade_pytorch_cuda12.1.sh
```

或手动安装：

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 步骤 5: 验证 PyTorch CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

### 方法 2: 手动安装

#### 步骤 1: 下载 CUDA 12.1

访问 NVIDIA 官网下载：
- https://developer.nvidia.com/cuda-12-1-0-download-archive

选择：
- Linux
- x86_64
- Ubuntu/Debian (根据你的系统)
- runfile (local)

#### 步骤 2: 安装 CUDA 12.1

```bash
# 下载 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# 运行安装（只安装 toolkit，不安装驱动）
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent --override
```

#### 步骤 3: 设置环境变量

编辑 `~/.bashrc` 或 `/etc/profile.d/cuda.sh`:

```bash
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

#### 步骤 4: 创建符号链接

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda
```

#### 步骤 5: 升级 PyTorch

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 验证安装

### 1. 验证 CUDA

```bash
nvcc --version
# 应该显示: release 12.1, V12.1.xxx
```

### 2. 验证 PyTorch

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

预期输出：
```
PyTorch version: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: [你的 GPU 名称]
```

---

## 常见问题

### 1. 驱动不兼容

如果驱动版本过低，需要先升级 NVIDIA 驱动：

```bash
# 检查驱动版本
nvidia-smi

# 如果需要，升级驱动
sudo apt update
sudo apt install nvidia-driver-580  # 或更新版本
sudo reboot
```

### 2. 多个 CUDA 版本共存

系统可以同时安装多个 CUDA 版本。通过符号链接 `/usr/local/cuda` 切换：

```bash
# 切换到 CUDA 12.1
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda

# 切换到 CUDA 11.8
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda
```

### 3. PyTorch 找不到 CUDA

确保：
1. CUDA 12.1 已正确安装
2. 环境变量已设置
3. PyTorch 版本支持 CUDA 12.1

```bash
# 检查环境变量
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda

# 重新安装 PyTorch
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

### 4. 编译错误

如果遇到编译错误，可能需要安装 CUDA 开发工具：

```bash
sudo apt install nvidia-cuda-toolkit
```

---

## 回退到 CUDA 11.8

如果需要回退：

```bash
# 1. 切换 CUDA 符号链接
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda

# 2. 更新环境变量
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 3. 重新安装 PyTorch (CUDA 11.8)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 参考链接

- [CUDA 12.1 下载页面](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 兼容性矩阵](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

---

## 注意事项

1. **备份重要数据**：升级前建议备份重要数据
2. **测试环境**：建议先在测试环境验证
3. **依赖检查**：确保所有依赖库支持 CUDA 12.1
4. **驱动兼容性**：确保 NVIDIA 驱动版本 >= 530.30.02（CUDA 12.1 要求）

---

**升级完成后，记得测试你的模型和代码是否正常工作！**






















