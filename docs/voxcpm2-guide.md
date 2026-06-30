# VoxCPM2 开发指南

## 简介

VoxCPM2 是一个 **2B 参数**的多语言、无 tokenizer 的 TTS 模型，输出 **48kHz** 录音室级音频。支持零样本生成、语音设计（通过文本描述创建声音）、声音克隆以及长文本续说。覆盖 **30 种语言**，包括中英文、日韩语、印尼语等。

- 原始 PyTorch 模型：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- MLX 社区转换版（可直接使用）：`mlx-community/VoxCPM2-<precision>`

---

## 快速开始

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/VoxCPM2-8bit")

result = next(model.generate("你好，这是 VoxCPM2 在 Apple Silicon 上运行。"))
audio = result.audio  # mlx array, 48kHz
```

---

## 核心功能

### 1. 语音设计（Voice Design）

无需参考音频，仅通过文字描述即可创造声音：

```python
result = next(model.generate(
    text="欢迎使用 VoxCPM2。",
    instruct="一位年轻女性，声音温暖而柔和",
))
```

`instruct` 参数支持的描述维度：
- 性别 / 年龄（年轻女性、中年男性、儿童）
- 音色特征（温暖、明亮、低沉、沙哑）
- 情感语气（温柔、严肃、欢快、平静）
- 场景风格（播音腔、耳语、旁白）

### 2. 声音克隆（Voice Cloning）

从一段音频样本中克隆声音：

```python
result = next(model.generate(
    text="这段文字将使用参考声音朗读。",
    ref_audio="speaker.wav",  # 参考音频路径
))
```

要求：
- 参考音频建议 **3-10 秒**，清晰无背景噪音
- 采样率不限，模型内部会自行处理

### 3. 长文续说（Continuation）

适合有声书、长篇小说等场景，无缝衔接：

```python
result = next(model.generate(
    text="前面的句子说到这里，",
    prompt_text="这是之前朗读的那句话。",
    prompt_audio="previous.wav",
))
```

工作原理：模型通过 `prompt_text` + `prompt_audio` 获取上下文，在此基础上继续生成 `text` 的语音，保持语速、音色和语气一致。

---

## 生成参数详解

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `inference_timesteps` | 10 | CFM 扩散步数 | 5-10：更快生成；15-20：更高质量 |
| `cfg_value` | 2.0 | 无分类器引导强度（CFG） | 1.5-3.0，越高越稳定但可能降低多样性 |
| `instruct` | `None` | 语音设计提示文本 | 仅在设计新声音时传入 |
| `ref_audio` | `None` | 参考音频路径或数组 | 克隆声音时传入 |
| `prompt_text` | `None` | 续说模式的提示文本 | 必须与 `prompt_audio` 配对使用 |
| `prompt_audio` | `None` | 续说模式的参考音频 | 必须与 `prompt_text` 配对使用 |
| `warmup_patches` | 0 | 额外生成的预热 patch 数 | 若开头不稳定可设为 1-3 |
| `max_tokens` | 2000 | 最大音频 patch 数 | 约对应 30-60 秒语音 |

---

## 可用模型

| 模型 ID | 精度 | 大小 | 适用场景 |
|---------|------|------|----------|
| `mlx-community/VoxCPM2-bf16` | bf16 | 4.96 GB | 最高质量，适合离线批量生成 |
| `mlx-community/VoxCPM2-8bit` | 8-bit | 3.23 GB | 平衡选择，推荐日常使用 |
| `mlx-community/VoxCPM2-4bit` | 4-bit | 2.30 GB | 速度优先，适合低显存设备 |

选择建议：在 Apple Silicon 上，8bit 版本是较好的平衡点。如果内存充足（>16GB）且追求质量，选 bf16。

---

## CLI 命令行

```bash
# 零样本生成并播放
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "你好世界" \
  --play

# 语音设计
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "你好世界" \
  --instruct "年轻女性，温馨的声音" \
  --play

# 声音克隆
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "你好世界" \
  --ref_audio speaker.wav \
  --ref_text "placeholder" \
  --play
```

---

## 架构概览

```
┌─────────────────────────────────────────────┐
│  MiniCPM4 Backbone                           │
│  (2048 hidden, 28 layers, GQA)              │
├─────────────────────────────────────────────┤
│  Residual LM (8 layers, 无 RoPE)             │
├─────────────────────────────────────────────┤
│  VoxCPMLocDiTV2                              │
│  (1024 hidden, 12 layers, multi-token mu)   │
│  + CFM diffusion sampling                   │
├─────────────────────────────────────────────┤
│  AudioVAE V2                                 │
│  (16kHz 编码 → 48kHz 解码)                   │
└─────────────────────────────────────────────┘
```

关键设计点：
- **无 tokenizer**：直接将文本/音频映射到连续表示，避免了离散 token 的信息损失
- **非对称编解码**：编码端 16kHz（更高效），解码端 48kHz（更高音质）
- **标量量化**：压缩音频 token，降低内存占用
- **CFM 扩散采样**：通过 `inference_timesteps` 控制质量/速度权衡

---

## 集成注意事项

1. **依赖**：需要 `mlx-audio` 库，确保已安装 `pip install mlx-audio`
2. **首次加载**：模型会自动从 Hugging Face 下载，约 3-5 GB
3. **内存**：8bit 版本约占用 4-6 GB 显存/内存；bf16 版本约 7-9 GB
4. **性能**：Apple Silicon M 系列芯片上，8bit 版本生成 10 秒音频约需 2-5 秒
5. **多语言**：模型会自动检测输入语言，无需手动指定

---

## License

Apache License 2.0
