# 黑屏问题诊断报告

## 问题确认

✅ **保存的帧图片确实是几乎全黑的**
- 平均亮度值：0.3（正常应该在50-200之间）
- 图片尺寸：1024x512（正常）
- 总帧数：564帧（正常）
- 所有检查的帧都是几乎全黑

## 问题分析

### 1. 数据流分析

从代码 `obs_to_image` 函数看，保存的图片包含两部分：
- **左侧**：机器人相机RGB图像（256x256，放大到512x512）
- **右侧**：俯视相机图像（256x256，放大到512x512）

### 2. 可能的原因

#### 原因A：相机没有正确获取图像数据
- 相机可能没有正确初始化
- 相机视角可能没有对准场景
- 场景可能没有正确加载

#### 原因B：图像数据格式问题
- RGB数据可能是0-1范围的float，但normalize可能有问题
- 数据可能是全0或接近全0

#### 原因C：渲染问题
- Isaac Sim渲染可能没有正确工作
- GPU渲染可能失败，但程序没有报错

## 诊断步骤

### 步骤1：检查相机数据原始值

在容器内运行：
```python
# 检查obs中的rgb和topdown_rgb数据
import numpy as np
# 需要在实际运行中打印obs数据
```

### 步骤2：检查相机配置

配置文件：`scripts/eval/configs/h1_internvla_n1_async_cfg.py`
- `camera_prim_path='torso_link/h1_1_25_down_30'` - 检查这个相机路径是否正确
- `camera_resolution=[640, 480]` - 检查分辨率

### 步骤3：检查场景加载

从日志看：
- 机器人位置：`[17.2064991, 2.06001997, 1.22162801]` ✓
- 场景应该已加载

### 步骤4：检查渲染日志

查看是否有渲染相关的错误或警告

## 建议的修复方案

### 方案1：检查相机数据（推荐）

在代码中添加调试输出，检查obs中的rgb数据：
```python
# 在 internnav/evaluator/utils/common.py 的 obs_to_image 函数中添加
print(f"RGB array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
print(f"RGB min: {rgb_array.min()}, max: {rgb_array.max()}, mean: {rgb_array.mean()}")
print(f"Topdown array shape: {topdown_array.shape}, dtype: {topdown_array.dtype}")
print(f"Topdown min: {topdown_array.min()}, max: {topdown_array.max()}, mean: {topdown_array.mean()}")
```

### 方案2：检查相机是否正确初始化

检查相机是否在USD文件中存在，路径是否正确

### 方案3：检查渲染模式

确认Isaac Sim是否正确渲染，可能需要：
- 检查headless模式设置
- 检查GPU渲染是否正常
- 检查Vulkan/OpenGL是否正常工作

## 下一步行动

1. **立即检查**：在容器内查看实际运行的日志，看是否有相机相关的错误
2. **添加调试**：在代码中添加打印语句，检查obs数据
3. **验证相机**：确认相机路径和配置是否正确
4. **检查渲染**：确认Isaac Sim渲染是否正常工作















