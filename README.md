# TTS-Serve-MLX

多模型文本转语音 API 服务（Qwen3-TTS + VoxCPM2），基于 Apple MLX 框架，在 Apple Silicon (M 系列芯片) 上本地运行。

## 功能

- **语音克隆** — 提供参考音频，克隆音色和情感风格
- **批量配音** — 多段音频批量生成，自动合并
- **对话生成** — 多角色对话，带交叉淡入淡出
- **基础 TTS** — 使用预置声音生成语音
- **批量 TTS** — 利用 GPU 批量推理，2-5 倍加速
- **声音设计** — 通过文字描述创建全新声音
- **情感控制** — 通过 `instruct` 参数控制情感风格
- **语音转文本** — Whisper 模型转录音频
- **流式生成** — 可选流式输出，降低首字节延迟

## 要求

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.14+
- 系统依赖: `brew install libsndfile`

## 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装构建工具（可选）
pip install -r requirements-dev.txt
```

## 模型

需要将模型文件放置于 `models/` 目录下：

- `models/qwenTTS_0.6B_MLX/` — Qwen3-TTS 0.6B 模型 (4-bit 量化)
- `models/whisper_asr_MLX/` — Whisper Large v3 Turbo ASR 模型 (FP16)

模型可通过 mlx-audio 工具转换或从 HuggingFace 下载。

## 启动服务

```bash
python3 server_main.py
```

服务默认在 `http://localhost:8000` 启动。API 文档: `http://localhost:8000/docs`

环境变量配置:
- `TTS_SERVE_PORT` — 端口号 (默认 8000)
- `TTS_SERVE_HOST` — 绑定地址 (默认 127.0.0.1)
- `TTS_SERVE_LOG_LEVEL` — 日志级别 (默认 warning)
- `TTS_SERVE_MODELS_DIR` — 模型目录 (默认 ./models)
- `TTS_SERVE_API_URL` — WebUI 连接的 API 地址 (默认 http://localhost:8000)

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/model-info` | 模型详情 |
| GET | `/model/status` | 模型加载状态 |
| POST | `/model/load` | 加载模型 (`tts` / `voxcpm2`) |
| POST | `/model/unload` | 卸载模型 |
| POST | `/clone` | 语音克隆 |
| POST | `/batch-clone` | 批量配音 |
| POST | `/dialogue` | 对话生成 |
| POST | `/tts` | 基础 TTS |
| POST | `/batch-tts` | 批量 TTS (加速) |
| POST | `/stt` | 语音转文本 |
| POST | `/voice-design` | 声音设计 |
| POST | `/vox/generate` | VoxCPM2 生成 |
| GET | `/files/{filename}` | 下载文件 |
| GET | `/files` | 文件列表 |

## 打包

```bash
./build.sh
```

产物输出到 `dist/qwen_tts_server/`，可用于 Electron 应用集成。

## 项目结构

```
├── api.py              # FastAPI 应用 (REST API)
├── tts_clone.py        # TTS 语音克隆核心模块
├── stt.py              # Whisper 语音转文本模块
├── server_main.py      # 服务启动入口
├── app.py              # 批量配音测试脚本
├── qwentts.py          # 最小化 TTS 示例
├── build.sh            # PyInstaller 打包脚本
├── requirements.txt    # 运行时依赖
├── requirements-dev.txt # 开发/构建依赖
└── models/             # 模型文件目录
```

## License

MIT
