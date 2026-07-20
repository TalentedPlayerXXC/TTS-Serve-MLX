# 📜 更新日志 ← 全是黑历史，慎翻

*更了就更，不更就不更，反正咕了你也拿我没办法 🐦*

---

## 2026-07-20 — 🧹 代码规范化 + 🔌 端口动态化 + 🛡️ 安全加固

### 🛡️ 安全防护：Origin/Referer 校验中间件

- **新增 `restrict_origin` 中间件** — 所有 POST/DELETE 等改状态请求，如果浏览器带了 `Origin` 或 `Referer` 但不是 localhost/127.0.0.1，直接返回 403
- **防止恶意网站「吃内存」** — 开着 Electron 时访问恶意网页，`<form>` / `fetch` 打到后端的 `/clone`、`/batch-clone` 等推理接口会被拦截
- **不影响正常使用** — curl / Python 脚本 / Electron 主进程不带来源头，放行；Electron 渲染进程 `file://` 带 `Origin: null`，也放行
- **Electron 零改动** — 纯后端中间件，前端不需要改任何代码
- **Electron 集成指南同步** — 加了安全说明章节，解释中间件的工作原理和兼容性

### 🧹 代码打扫除

- **内联 import 全部提到模块顶部** — `import gc`、`import mlx.core as mx`、`from mlx_audio.tts.utils import load_model` 在 api.py 和 tts_clone.py 中各出现 3-4 次重复加载，现在统一提到文件顶部，清爽了 ✨
- **删除 `TTSItem` 死代码** — 这个数据类定义了以后从未被调用过，送走 ☠️
- **`generate_dialogue()` 标记 deprecated** — api.py 的 `/dialogue` 端点早已自己实现逐条循环，`TTSClone` 上的这个方法不再被调用，加了个注释标明
- **删掉 `api.py` 末尾的重复入口** — `if __name__ == "__main__"` + `uvicorn.run` 和 `server_main.py` 功能完全重复，统一走 `server_main.py`

### 🔌 后端端口动态化

- **端口不再写死 8000** — 没设 `TTS_SERVE_PORT` 时自动扫 8000-8050 的空闲端口，找到了就用
- **stdout 打印 `TTS_SERVER_PORT=xxxxx`** — Electron 主进程正则解析此值，从此不担心 8000 被占瘫痪
- **Electron 集成指南同步更新** — port 从环境变量改为 stdout 解析 + waitForPortAndHealth

### 🛡️ 安全小补丁

- **路径遍历防护补漏** — `_validate_audio_path()` 加 `os.path.normpath()` 预处理，防止 `foo/../../etc/passwd` 类路径绕过检查

### 🔊 音频后处理补全

- **模型原生 batch 路径添加淡入淡出** — 之前逐条生成路径有 `apply_fade`，batch 路径跳过了，行为不一致。现在 batch 路径也有了，首尾不爆音
- **VoxCPM2 输出添加淡入淡出** — 48kHz 双倍采样点，`save_file=true` 和 `save_file=false` 两条路径都补上了

### 🏗️ 构建 & 配置

- **`.spec` optimize=0 → optimize=2** — 打包优化级别拉满，产物更小
- **`build.sh` libsndfile 动态路径查找** — 同时支持 Intel（`/usr/local/lib`）和 ARM（`/opt/homebrew/lib`）Homebrew
- **配置去重** — `build.sh` 不再写一遍 `--hidden-import` / `--collect-all`，全部走 `tts_serve_mlx.spec`，两份不同步的风险消除
- **`by_size` 清理算法微调** — 之前第一个撑爆上限的文件会连带后面所有旧文件一起杀，现在逐个删直到低于上限，不会多删

### 🧼 其他小修

- **`.gitignore` 清理** — 删掉已不存在的 `Custom_MLX` 和 `whisper_asr_MLX` 残留意指
- **`server_main.py` 启动日志** — 启动时打印 `models_dir` 配置值，方便排查
- **`docs/voxcpm2-guide.md` 补充说明** — 加了段注释为什么项目选 4-bit 而非推荐的 8-bit（16GB M1 内存更稳）
- **Electron 集成指南轮询退避建议** — 大型模型下载 2GB+ 可考虑 1s→2s→4s 指数退避

### 📦 打包加固 & SSL 兼容

- **`certifi` 接管 SSL 验证** — 模型下载改用 `certifi.where()` 获取独立 CA 证书包，打包进 app 内，不依赖用户系统钥匙串
- **SSL 失败三层兜底** — `certifi` CA → `_create_unverified_context()` 降级，刚出厂的 Mac 也能正常下载模型
- **`build.sh` 首行 `unset PYTHONPATH`** — 防止 Hermes 3.11 包污染 PyInstaller 打包，pip 和 PyInstaller 都只从项目 venv 取包
- **`certifi` 强制安装到项目 venv** — 之前蹭 Hermes 3.11 的 certifi，打包后路径不可用，现在项目 venv 独立一份

### 📥 模型下载改用 HTTP 流式下载

- **`modelscope CLI` 替换为 `urllib` HTTP 流式下载** — 打包后的环境没有 `modelscope` 命令，改用 Python 标准库 `urllib.request`，0 依赖，打包后 100% 可用
- **按文件列表逐个下载** — 直接从 `_MODEL_SOURCES.files` 拿文件清单，通过 `resolve/main` 接口下载实际权重（不走 git LFS，不会被指针文件坑）
- **线程内异常兜底** — 所有异常被 try/except 捕获并更新到 `_download_tasks`，不会静默死亡

### 📊 进度算法修正

- **按字节加权计算整体进度** — 之前按文件个数平均算，model.safetensors 下了 91% 但整体只显示 7%，前后端对不上。改成已下字节/总字节，进度条和 message 一致
- **Electron 对接文档同步** — 描述从「git clone」改为「HTTP 流式下载」

---

## 2026-07-15 — 📥 模型下载接口优化

- **`GET /models-info` key 与文件夹名对齐** — `qwen3-tts` → `qwenTTS_0.6B_MLX`，`voxcpm2` → `voxCPM2_4bit_MLX`，前端无需映射表
- **新增 `files` 字段** — 每个模型列出所有需下载的文件，版本更新也不怕
- **Electron 集成指南同步** — 下载函数改为从 API 拿文件列表，不再硬编码

### 📥 后端一键下载

- **新增 `POST /model/download`** — 后端通过 `urllib` HTTP 流式下载模型，纯标准库 0 依赖，打包后 100% 可用
- **新增 `GET /model/download/status/{model}`** — 轮询进度接口，实时显示下载百分比
- **魔搭优先** — 默认走魔搭源，国内无需代理
- **后台异步下载** — POST 立即返回，前端轮询进度，不阻塞
- **自动跳过已下载** — 已下载的模型直接返回 `already_downloaded`，不会重复下载
- **同步清理** — 删除 `api.py` 中不再使用的 `ASR_MODEL_PATH`
- **修正魔搭地址** — Qwen3-TTS: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`，VoxCPM2: `mlx-community/VoxCPM2-4bit`

> 方案转变：原先设计由 **Electron 前端** HTTP 流式下载，改为**后端**一行 `modelscope download` 搞定。

### 🛡️ 修复模型加载失败变砖

- **`load_qwen3()` 改为先加载后赋值** — 模型加载失败时不再错误赋值全局变量，下次可重试
- **`load_vox()` 同理** — 同一个修复模式
- **`_download_tasks` 自动清理** — 下载完成/失败 30 秒后自动清理 entry，防止慢泄漏

### 🧠 加载新模型前自动卸载旧模型

- **`/model/load` 改为先卸再载** — 加载 TTS 前自动卸载 VoxCPM2，加载 VoxCPM2 前自动卸载 TTS
- 防止两个模型共存 GPU 导致内存爆炸（之前切换 VoxCPM2 能冲到 7GB+）

### 🧹 GPU 内存彻底释放

- **`mx.set_cache_limit(0)` + `mx.clear_cache()`** — 卸载模型时禁用 MLX 缓存，强制释放 Metal 缓冲区
- 之前只调 `clear_cache()`，MLX 把缓冲区留在池子里不还，反复切换模型导致 Metal 堆碎片化膨胀到 8.3GB
- 验证：active内存每次卸载归零，反复切换 3 次不涨

### 🐛 修复批量配音 & 对话

- **原生 batch 路径已冻结** — `batch_generate()` 要求 `ref_audio` + `ref_text` 必须同时传，Speaker 模式走这条路直接抛异常，已加条件拦截
- **`/dialogue` 重写** — 干掉 `TTSClone.batch_generate()` 包装层，改为逐条直接调 `generate()`，解决第二次请求卡死
- **预校验路径** — dialogue 现在和 batch-clone 一样先检查所有音频路径再生成
- **新增 `silence_duration` 参数** — 默认 0.3s，对话段落间可自定义间隔

---

## 2026-07-14 — 🗑️ 送走 STT + ⚡ 批量配音优化

### 🗑️ 送走 STT

Whisper STT 正式退役。ICL 模式已弃用，STT 再无用处，直接送走。

- 删除 `stt.py`
- `api.py` 彻底清理：移除 `stt` 全局变量、load/unload、`/stt` 端点、`ModelLoadRequest` 中 `stt` 选项
- `build.sh` 移除 STT 相关 hidden-import 和模型目录
- 文档全面清理：README / api.md / ELECTRON_INTEGRATION.md
- 3个模型 → 2个模型（Qwen3-TTS + VoxCPM2），清爽多了

### ⚡ 批量配音性能优化

`/batch-clone` 端点重写，三项优化：

1. **不存中间文件** — `merge=True` 时每段不再单独写磁盘，全在内存中搞定
2. **预校验路径** — 先一次检查所有音频路径再开始生成，避免生成到一半报错
3. **同角色走模型原生 batch** — N 段文本共用同一个 `ref_audio` 时，自动走 Qwen3-TTS 的 `batch_generate()`，一次前向处理所有文本

前端完全不用改，传参和返回值兼容。

---

## 2026-07-08 — 🧼 卸载释放资源

修复了模型卸载时 GPU 内存未释放的问题：

- `unload_qwen3()` / `unload_vox()` 统一加上 `mx.clear_cache()`
- `TTSClone.unload()` 同样加上 `mx.clear_cache()`
- 现在卸载链路完整：**删引用 → gc.collect() → MLX 清理 Metal 缓存**
- 验证：两个模型均能正常加载→卸载→重新加载→再次正常生成

---

## 2026-07-06 — 💥 架构大重构：三兄弟分家了

> **前情提要**：经过 N 轮 AB 测试、无数次「效果很差劲」的暴击、
> 以及 Speaker+instruct 余弦相似度暴跌到 0.90 的惨痛教训，
> 终于找到了正确的打开方式。

### 🧠 最终模型分工

```
Qwen3-TTS → Speaker 模式（快！省！无 ASR！）
VoxCPM2   → 情感克隆 + 声音设计（steps=6, cfg=4.0 真香）
```

### 🔥 到底改了啥

#### 2026-07-06 第二轮 — 缓存管理上线

新增两个接口：
- `GET /cache` — 查看缓存状态（文件数、大小、最旧/最新）
- `POST /cleanup` — 三种清理模式：all / older_than / by_size

#### `tts_clone.py` — 瘦身成功

删掉历史的 ASR 依赖——`asr_model` 参数、自动 STT 逻辑。`generate()` 方法 `ref_text` 不传就自动走 Speaker Embedding 模式，不再试图用 Whisper 去识别参考音频文本。

#### `api.py` — 大扫除

- 删掉 `_inject_asr_to_tts()`、`_qwen_asr`、`_asr_loaded_at`，这些历史遗留代码早该入土了
- 新增 `_stt_model` + `load_stt()` — STT 独立了，自由了！
- `ModelLoadRequest` 现在也接受 `"stt"` 作为合法值，前端想用 STT 就自己加载
- `/model/unload` 和状态端点同步支持 `stt`
- VoxCPM2 默认参数修正为 steps=6, cfg=4.0

#### 文档同步更新

- `api.md` — 记录了新架构和 Speaker 模式用法
- `api.py` — docstring 和注释全部跟上

> 从代码到文档，全链路清理了一遍，舒坦。

---

*下次更新不知道是什么时候，随缘。*
