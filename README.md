# 🗣️ TTS-Serve-MLX

> **⚠️ 免责声明**：本项目仅供学习研究，**严禁用于任何违法用途**（包括但不限于诈骗、伪造身份、侵犯他人肖像权/声音权等）。玩归玩，别越线，进去了我可捞不了你 🙏

> 让你的 Apple Silicon Mac **一秒变身声优工作室**。
> 克隆声音、批量配音、情感克隆、声音设计——全在本地跑，不上云，**不花钱**。
>
> 基于 **Qwen3-TTS** + **VoxCPM2** 的双模型语音服务，Apple MLX 框架驱动，M 系列芯片专属。Intel？不存在的。

> 📢 **更新公告**：想知道每次改了啥？→ [`UPDATE.md`](UPDATE.md)，黑历史全在里面 👀

## 🎪 它能干啥

| 技能 | 谁干的 | 有多强 |
|------|--------|--------|
| 🎭 **语音克隆** | Qwen3-TTS（Speaker 模式） | 给段音频就学你说话，快得不讲道理 |
| 🧵 **批量配音** | Qwen3-TTS | 一次塞十几段，默默肝完还帮你拼好 |
| 💬 **对话生成** | Qwen3-TTS | 多角色唠嗑，自动加停顿，像模像样 |
| 🎨 **声音设计** | VoxCPM2 | 输入「沉稳大叔音」，它真给你捏一个 |
| 🎭 **情感克隆** | VoxCPM2 [steps=6, cfg=4.0] | 克隆音色 + 情绪指令，嬉笑怒骂随你 |
| 👂 ~~语音转文本~~ | ~~Whisper（已移除）~~ | ~~不再需要，ICL 已弃用~~ |
| ~~⚡ 流式生成~~ | 都支持 | 边生边播，不等进度条（TODO：未充分测试） |

## 🧠 模型分工

玩明白了之后，三兄弟各司其职：

```
Qwen3-TTS（Speaker 模式）→ 日常苦力：克隆、批量、对话
               ├── 快！不用 ASR！
               ├── 内置 speaker_encoder 直接提取音色
               └── 想玩花的？传 ref_text 切 ICL 模式

VoxCPM2 [steps=6, cfg=4.0]  → 情感担当：情绪克隆、声音设计
               ├── 扩散模型，原生 instruct 支持
               ├── 输入「开心」它真开心
               └── 就是有点慢，但慢得值
```

---

## 🛠️ 准备工作

- macOS（Apple Silicon M1/M2/M3/M4，**Intel 退散** ❌）
- Python 3.14+
- 系统依赖：`brew install libsndfile`

## 📦 装起来

```bash
# 创建虚拟环境（好习惯）
python3 -m venv .venv
source .venv/bin/activate

# 装依赖（全部在一个文件里）
pip install -r requirements.txt
```

## 📁 模型文件

把模型们请到 `models/` 目录下：

```
models/
├── qwenTTS_0.6B_MLX/     ← Qwen3-TTS 0.6B（4-bit 量化，主力牛马）
└── voxCPM2_4bit_MLX/      ← VoxCPM2 2B（4-bit 量化，扩散模型，情感专家）
```

## 🚀 启动！

```bash
python3 server_main.py
```

服务默认蹲在 `http://localhost:8000`，Swagger 文档在 `http://localhost:8000/docs`。

可以捏几个环境变量调教它：

| 变量 | 默认值 | 干嘛的 |
|------|--------|--------|
| `TTS_SERVE_PORT` | `8000`（未设时自动扫描 8000-8050） | 端口号 |
| `TTS_SERVE_HOST` | `127.0.0.1` | 绑哪，别乱 expose |
| `TTS_SERVE_LOG_LEVEL` | `warning` | 想听它叨叨就改 info |
| `TTS_SERVE_MODELS_DIR` | `./models` | 模型藏哪了 |

## 🔌 API 端点一览

模型默认**在睡觉**，先 `POST /model/load` 喊它起床。

| 方法 | 路径 | 说人话的说明 |
|------|------|-------------|
| `GET` | `/health` | 还活着吗？模型醒了吗？ |
| `GET` | `/model-info` | 模型们都住哪、醒着没？ |
| `GET` | `/model/status` | 醒了没？简洁版 |
| `POST` | `/model/load` | 喊起床：`tts`（Speaker）、`voxcpm2`（情感） |
| `POST` | `/model/unload` | 让模型回去睡，省内存 |
| `POST` | `/model/download` | 📥 一键下载模型，后端 HTTP 流式下载，魔搭优先 |
| `POST` | `/clone` | 🎭 语音克隆 — Speaker 模式速通，传 ref_text 可切 ICL |
| `POST` | `/batch-clone` | 📦 批量配音 — 一次塞 N 段，可选合并 |
| `POST` | `/dialogue` | 💬 对话 — 多角色唠嗑，自动拼接 |
| `POST` | `/vox/clone` | 🎤 VoxCPM2 情感克隆 — 扩散模型 + 情绪指令，默认 steps=6, cfg=4.0 |
| `POST` | `/vox/design` | 🎨 VoxCPM2 声音设计 — 用文字捏声音，默认 steps=6, cfg=4.0 |
| `GET` | `/files/{filename}` | ⬇️ 下载生成的音频 |
| `GET` | `/files` | 📋 列出所有生成的文件（支持翻页） |
| `GET` | `/cache` | 📊 查看缓存状态（文件数、大小、最旧/最新） |
| `POST` | `/cleanup` | 🧹 清理缓存：`all` 清空、`older_than` 按小时、`by_size` 按大小 |

> 所有生成的音频都在 `./api_output/` 里，也可以用 `/output/{文件名}` 直接薅走。

## 🏃 快速上手

```bash
# 1. 喊模型起床
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "tts"}'

# 2. 克隆一段声音（Speaker 模式，不用传 ref_text）
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

# 4. 下载生成的音频
curl -O http://localhost:8000/output/clone_xxxx.wav
```

## 📦 打包带走

```bash
./build.sh
```

产物在 `dist/qwen_tts_server/`，可以塞进 Electron 里做个桌面应用。Electron 集成指南看 `ELECTRON_INTEGRATION.md`。

## 🗂️ 项目骨架

```
├── api.py                  ← FastAPI 主应用，所有接口在这
├── tts_clone.py            ← Qwen3 语音克隆（Speaker 模式）
├── server_main.py          ← 服务启动入口
├── build.sh                ← PyInstaller 打包脚本
├── api.md                  ← 详细 API 文档（正经版）
├── ELECTRON_INTEGRATION.md ← Electron 集成指南
├── UPDATE.md               ← 更新日志
└── models/                 ← 模型住的地方
```

## 📄 License

MIT — 随便玩，欢迎 PR，提了必看 👀

## 🙏 致谢

感谢以下游戏的配音演员们提供了宝贵的参考音频素材——

| 游戏 | 代表角色 |
|------|---------|
| 🌌 **星穹铁道** | 花火、三月七、停云、银狼 |
| 🏔️ **原神** | 七七、胡桃、甘雨 |
| ⚡ **崩坏3** | 人之律者（爱莉希雅） |
| 🎮 **绝区零** | 铃、星见雅 |
| 🌊 **鸣潮** | 今汐、椿 |

想知道每次更新改了啥？→ `UPDATE.md`，黑历史全在里面 👀
