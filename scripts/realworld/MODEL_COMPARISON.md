# InternVLA-N1 模型对比分析

## 本地模型 vs Hugging Face 官方模型

### 基本信息对比

| 项目 | 本地模型 | Hugging Face (InternVLA-N1-wo-dagger) |
|------|---------|--------------------------------------|
| **模型路径** | `checkpoints/InternVLA-N1` | `InternRobotics/InternVLA-N1-wo-dagger` |
| **模型类型** | `internvla_n1` | `internvla_n1` |
| **架构** | `InternVLAN1ForCausalLM` | `InternVLAN1ForCausalLM` |
| **模型大小** | ~15.6 GB (4个分片) | ~8B 参数 (BF16) |
| **状态** | 需要确认版本 | 官方稳定版本 (推荐) |

### 配置参数对比

#### 本地模型配置 (`config.json`)

```json
{
  "model_type": "internvla_n1",
  "hidden_size": 3584,
  "vocab_size": 151668,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "intermediate_size": 18944,
  "vision_config": {
    "hidden_size": 1280,
    "depth": 32,
    "num_heads": 16
  },
  "navdp_version": 0.1,
  "n_query": 16
}
```

#### Hugging Face 模型信息

根据 [Hugging Face 页面](https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger)：
- **模型大小**: 8B 参数
- **Tensor 类型**: BF16
- **Chat template**: 支持
- **License**: CC-BY-NC-SA-4.0

### 版本差异

根据 Hugging Face 文档，存在两个版本：

#### InternVLA-N1-Preview (预览版)
- **系统设计**: 双系统（同步）
- **训练**: System 1 仅在 System 2 推理步骤时训练
- **推理**: System 1、2 以相同频率推理 (~2 Hz)
- **状态**: 历史预览版

#### InternVLA-N1 (官方版，wo-dagger)
- **系统设计**: 双系统（异步）✅
- **训练**: System 1 在更密集步长 (~25 cm) 上训练，使用最新 System 2 隐状态 ✅
- **推理**: System 1、2 异步推理，允许动态避障 ✅
- **性能**: 改进的平滑性、效率和真实世界零样本泛化 ✅
- **状态**: 稳定官方发布版（推荐）✅

### 文件结构对比

#### 本地模型文件
```
checkpoints/InternVLA-N1/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors (4.7 GB)
├── model-00002-of-00004.safetensors (4.7 GB)
├── model-00003-of-00004.safetensors (4.6 GB)
├── model-00004-of-00004.safetensors (1.8 GB)
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json
├── chat_template.json
└── README.md
```

#### Hugging Face 模型文件
应该包含类似的文件结构，但需要确认：
- 模型权重文件（safetensors）
- 配置文件
- Tokenizer 文件
- Chat template

### 生成配置对比

#### 本地模型 (`generation_config.json`)
```json
{
  "do_sample": true,
  "temperature": 0.1,
  "top_k": 1,
  "top_p": 0.001,
  "repetition_penalty": 1.05,
  "attn_implementation": "flash_attention_2"
}
```

**注意**: 本地模型的 `repetition_penalty` 设置为 1.05，这可能不足以防止重复生成问题。

### 关键发现

1. **模型版本不确定**: 
   - 本地模型可能是 Preview 版本或早期版本
   - 需要确认是否与官方最新版本一致

2. **配置差异**:
   - 本地模型有完整的配置文件
   - 需要对比 Hugging Face 上的实际配置

3. **生成问题**:
   - 测试发现模型在纯文本输入时生成重复感叹号
   - 这可能与模型版本或配置有关

### 建议

1. **验证模型版本**:
   ```bash
   # 检查模型文件的创建/修改时间
   ls -lh checkpoints/InternVLA-N1/*.safetensors
   
   # 对比配置文件
   diff checkpoints/InternVLA-N1/config.json <(从HF下载的config.json)
   ```

2. **如果版本不一致，建议更新**:
   ```python
   from transformers import AutoModelForCausalLM, AutoProcessor
   
   model = AutoModelForCausalLM.from_pretrained(
       "InternRobotics/InternVLA-N1-wo-dagger",
       torch_dtype=torch.bfloat16,
       device_map="auto"
   )
   ```

3. **优化生成参数**:
   - 增加 `repetition_penalty` 到 1.1-1.2
   - 确保提供图像输入（模型是视觉语言模型）

### 参考链接

- [Hugging Face 模型页面](https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger)
- [项目主页](https://internrobotics.github.io/internvla-n1.github.io/)
- [技术报告](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)
- [GitHub 代码库](https://github.com/InternRobotics/InternNav)

