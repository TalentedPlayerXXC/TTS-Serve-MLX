# 🗣️ TTS-Serve-MLX

> 让你的 Apple Silicon Mac 一秒变身声优工作室。
> 克隆声音、批量配音、识别语音——全在本地跑，不上云，不花钱。

基于 **Qwen3-TTS** + **Whisper ASR** + **VoxCPM2** 的多模型语音服务，Apple MLX 框架驱动，M 系列芯片专属。

## 它能干啥

- **🎭 语音克隆** — 给一段参考音频，它就能模仿着说出你想说的话
- **🧵 批量配音** — 一次塞十几段，它默默肝完还帮你拼好
- **💬 对话生成** — 多个角色唠嗑，自动加淡入淡出，像模像样
- **🎨 VoxCPM2 声音设计** — 输入「沉稳大叔音」，它真给你捏一个
- **🎨 VoxCPM2 情感克隆** — 克隆音色的同时还能加情绪指令，嬉笑怒骂随你
- **👂 语音转文本** — Whisper 帮你听写，省了打字的手指
- **⚡ 流式生成** — 边生成边播，不用傻等进度条

## 准备工作

- macOS（Apple Silicon M1/M2/M3/M4，Intel 退散）
- Python 3.14+
- 系统依赖：`brew install libsndfile`

## 装起来

```bash
# 创建虚拟环境（好习惯）
python3 -m venv .venv
source .venv/bin/activate

# 装依赖
pip install -r requirements.txt

# 如果要打包成独立应用（可选）
pip install -r requirements-dev.txt

# 如果要 WebUI 界面（可选）
pip install -r requirements-webui.txt
```

## 模型文件

把模型们请到 `models/` 目录下：

```
models/
├── qwenTTS_0.6B_MLX/     ← Qwen3-TTS 0.6B（4-bit 量化，主力干将）
├── whisper_asr_MLX/       ← Whisper Large v3 Turbo ASR（顺风耳）
└── voxCPM2_4bit_MLX/      ← VoxCPM2 2B（4-bit 量化，扩散模型，声音设计专属）
```

## 启动！

```bash
python3 server_main.py
```

服务默认蹲在 `http://localhost:8000`，Swagger 文档在 `http://localhost:8000/docs`。

可以捏几个环境变量调教它：

| 变量 | 默认值 | 干嘛的 |
|------|--------|--------|
| `TTS_SERVE_PORT` | `8000` | 端口号 |
| `TTS_SERVE_HOST` | `127.0.0.1` | 绑定的地址 |
| `TTS_SERVE_LOG_LEVEL` | `warning` | 日志叨叨程度 |
| `TTS_SERVE_MODELS_DIR` | `./models` | 模型放哪 |
| `TTS_SERVE_API_URL` | `http://localhost:8000` | WebUI 连哪个 API |
| `TTS_SERVE_AUTO_START_API` | `1` | WebUI 要不要自动拉起 API |

## API 端点一览

模型默认**不预加载**，先 `POST /model/load` 喊它起床。

| 方法 | 路径 | 说人话的说明 |
|------|------|-------------|
| `GET` | `/health` | 还活着吗？模型醒了吗？ |
| `GET` | `/model-info` | 模型们都住哪、醒着没？ |
| `GET` | `/model/status` | 模型醒了没？（简洁版） |
| `POST` | `/model/load` | 喊模型起床：`{"model": "tts"}` 或 `{"model": "voxcpm2"}` |
| `POST` | `/model/unload` | 让模型回去睡 |
| `POST` | `/clone` | 🎭 语音克隆 — 给参考音频，念你写的词 |
| `POST` | `/batch-clone` | 📦 批量配音 — 一次提多个任务，可选合并成一个文件 |
| `POST` | `/dialogue` | 💬 对话生成 — 多角色轮流说话，自动拼接 |
| `POST` | `/stt` | 👂 语音转文本 — 给音频，还你文字 |
| `POST` | `/vox/clone` | 🎤 VoxCPM2 克隆 — 带情感控制的扩散模型克隆 |
| `POST` | `/vox/design` | 🎨 VoxCPM2 声音设计 — 用文字描述捏一个声音 |
| `GET` | `/files/{filename}` | ⬇️ 下载生成的音频文件 |
| `GET` | `/files` | 📋 列出所有生成的文件（支持分页） |

> 所有生成的音频都在 `./api_output/` 里，也可以用 `/output/{文件名}` 直接薅走。

## 快速上手

```bash
# 1. 让模型起床
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "tts"}'

# 2. 克隆一段声音
curl -X POST http://localhost:8000/clone \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "ref_audio": "./demo.wav"}'

# 3. 让 VoxCPM2 捏个大叔音
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "voxcpm2"}'

curl -X POST http://localhost:8000/vox/design \
  -H "Content-Type: application/json" \
  -d '{"text": "欢迎收听新闻", "instruct": "沉稳的中年男声，语速适中"}'

# 4. 语音转文字
curl -X POST http://localhost:8000/stt \
  -H "Content-Type: application/json" \
  -d '{"ref_audio": "./speech.wav"}'

# 5. 下载生成的音频
curl -O http://localhost:8000/output/clone_xxxx.wav
```

## 还有 WebUI？

对，附赠一个 Gradio 网页界面：

```bash
pip install -r requirements-webui.txt
python3 webui.py
```

然后打开 `http://localhost:7860`，点鼠标就能玩。

## 打包带走

```bash
./build.sh
```

产物在 `dist/qwen_tts_server/`，可以塞进 Electron 里做个桌面应用。

## 项目骨架

```
├── api.py              ← FastAPI 主应用，所有接口在这
├── tts_clone.py        ← Qwen3 语音克隆核心
├── stt.py              ← Whisper 语音识别
├── server_main.py      ← 服务启动入口
├── webui.py            ← Gradio 图形界面
├── app.py / qwentts.py / vox_test.py  ← 各种测试/示例脚本
├── build.sh            ← PyInstaller 打包脚本
├── api.md              ← 详细 API 文档
└── models/             ← 模型住的地方
```

## License

MIT — 随便玩，欢迎 PR。
