# 📜 更新日志 ← 全是黑历史，慎翻

*更了就更，不更就不更，反正咕了你也拿我没办法 🐦*

---

## 2026-07-15 — 📥 模型下载接口优化

- **`GET /models-info` key 与文件夹名对齐** — `qwen3-tts` → `qwenTTS_0.6B_MLX`，`voxcpm2` → `voxCPM2_4bit_MLX`，前端无需映射表
- **新增 `files` 字段** — 每个模型列出所有需下载的文件，版本更新也不怕
- **Electron 集成指南同步** — 下载函数改为从 API 拿文件列表，不再硬编码

### 📥 后端一键下载

- **新增 `POST /model/download`** — 后端通过 `modelscope download` 或 `huggingface-cli` 直接下载模型
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
Whisper   → 独立 STT，谁也别绑我
```

### 🔥 到底改了啥

#### 2026-07-06 第二轮 — 缓存管理上线

新增两个接口：
- `GET /cache` — 查看缓存状态（文件数、大小、最旧/最新）
- `POST /cleanup` — 三种清理模式：all / older_than / by_size

#### `tts_clone.py` — 瘦身成功
- 砍掉了 `asr_model` 参数（再也不给 TTS 绑个累赘了）
- 砍掉了 `generate()` 里的自动 ASR 转录（再也不求 Whisper 了）
- `ref_text=None` → Speaker Embedding 模式，Qwen 自己搞定
- `ref_text=xxx` → ICL 模式，大佬请随意

#### `api.py` — 大扫除
- 删了 `_qwen_asr`、`_asr_loaded_at`、`_inject_asr_to_tts()` ⚰️ RIP
- `load_qwen3()` 从「买一送一搭个 Whisper」变成「单独一个 TTS」
- 新增 `_stt_model` + `load_stt()` — STT 独立了，自由了！
- `/model/load` 现在支持 `{"model": "stt"}`，想用才加载
- `/stt` 端点翻身做主人，不依赖 TTS 了
- 状态端点 `whisper` → `stt`（改名改命）

#### 参数变更（敲黑板！）

| 端点 | 参数 | 以前 | 现在 |
|------|------|------|------|
| `/vox/clone` | 步数 | 5 | **6** 🎯 |
| `/vox/clone` | CFG | 3.0 | **4.0** 🎯 |
| `/vox/design` | 步数 | 7 | **6** 🎯 |
| `/vox/design` | CFG | 3.0 | **4.0** 🎯 |

> 为什么是 [6, 4.0]？我们拿花火、停云、三月七的声音一个个试过来的，
> 试到耳朵起茧子才找到这个甜点参数。信我，好用。

### 🆕 新来的

| 文件 | 干嘛的 |
|------|--------|
| `UPDATE.md` | 就是你现在看的这个 |
| 若干测试脚本 | `asr_test.py`, `vox_test.py`… 调参调麻了，均已清理 |

### ✅ 验证结果

- API 路由 16 个全部正常加载 ✅
- 四人群聊 Speaker 模式测试通过 ✅
- VoxCPM2 [6, 4.0] 情感克隆通过 ✅
- VoxCPM2 声音设计通过 ✅
- STT 独立加载通过 ✅
- **Qwen ICL + initial_prompt 对不同声音乱加语气词** ❌ → 已弃用
- **Speaker + instruct 余弦相似度暴跌** ❌ → 已弃用

### 🧹 清理

- 删除 `webui.py`、`ab_test_*.py`、`vox_*.py`、`asr_test.py` 等测试脚本
- 合并 `requirements-dev.txt` 进 `requirements.txt`
- 删除 `requirements-webui.txt`

---

