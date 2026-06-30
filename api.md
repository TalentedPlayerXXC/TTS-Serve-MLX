# TTS-Serve API 文档

基于 FastAPI 的多功能语音服务，提供语音克隆、批量配音、语音转文本（STT）功能。

- **基础 URL**: `http://localhost:8000`
- **API 文档**: `http://localhost:8000/docs` (Swagger UI)
- **音频输出格式**: WAV
- **采样率**: Qwen3 TTS: 24000 Hz, VoxCPM2: 48000 Hz
- **输出目录**: `./api_output/`（通过 `/output/` 静态挂载访问）

---

## 目录

1. [模型管理](#1-模型管理)
2. [语音克隆](#2-语音克隆)
3. [批量配音](#3-批量配音)
4. [对话生成](#4-对话生成)
5. [语音转文本 (STT)](#5-语音转文本-stt)
6. [VoxCPM2 生成](#6-voxcpm2-生成)
7. [文件管理](#7-文件管理)
8. [错误码](#8-错误码)

---

## 1. 模型管理

模型默认**不预加载**，需先调用 `/model/load` 加载后再使用其他接口。

### 1.1 健康检查

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "qwen3_loaded": false,
  "whisper_loaded": false,
  "voxcpm2_loaded": false
}
```

### 1.2 模型信息

```
GET /model-info
```

**Response:**
```json
{
  "qwen3": {
    "path": "./models/qwenTTS_0.6B_MLX",
    "loaded": false,
    "has_asr_injected": false
  },
  "whisper": {
    "path": "./models/whisper_asr_MLX",
    "loaded": false
  },
  "voxcpm2": {
    "path": "./models/voxCPM2_4bit_MLX",
    "loaded": false
  }
}
```

### 1.3 模型加载状态

```
GET /model/status
```

**Response:**
```json
{
  "qwen3": false,
  "whisper": false,
  "voxcpm2": false
}
```

### 1.4 加载模型

```
POST /model/load
```

**Request Body:**
```json
{
  "model": "tts"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称，可选 `"tts"`（加载 Qwen3 TTS + Whisper ASR）或 `"voxcpm2"` |

**Response:**
```json
{
  "success": true,
  "model": "tts",
  "action": "loaded"
}
```

### 1.5 卸载模型

```
POST /model/unload
```

**Request Body:**
```json
{
  "model": "tts"
}
```

**Response:**
```json
{
  "success": true,
  "model": "tts",
  "action": "unloaded"
}
```

---

## 2. 语音克隆

使用参考音频克隆音色生成语音。

### 2.1 单条语音克隆

```
POST /clone
```

**Request Body:**
```json
{
  "text": "欢迎使用语音克隆系统",
  "ref_audio": "./audio/reference.wav",
  "ref_text": "这是参考音频的文本",
  "stream": false,
  "save_file": true,
  "filename": "my_clone"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 目标文本，1-5000 字符 |
| `ref_audio` | string | 是 | - | 参考音频文件路径 |
| `ref_text` | string | 否 | null | 参考音频的原文，若不提供则自动用 Whisper STT 识别 |
| `stream` | bool | 否 | false | 是否使用流式生成 |
| `save_file` | bool | 否 | true | 是否保存为文件。`false` 时直接返回 WAV 二进制流 |
| `filename` | string | 否 | null | 自定义文件名（不含扩展名） |

**Response (200, save_file=true):**
```json
{
  "success": true,
  "text": "欢迎使用语音克隆系统",
  "ref_audio": "./audio/reference.wav",
  "audio_url": "/output/clone_abc12345.wav",
  "filename": "clone_abc12345.wav"
}
```

> `save_file=false` 时返回 `audio/wav` 二进制流，不含 JSON。

---

## 3. 批量配音

### 3.1 批量语音克隆

```
POST /batch-clone
```

**Request Body:**
```json
{
  "items": [
    {
      "text": "第一段配音内容",
      "ref_audio": "./audio/ref1.wav",
      "ref_text": "参考音频文本一"
    },
    {
      "text": "第二段配音内容",
      "ref_audio": "./audio/ref2.wav",
      "ref_text": "参考音频文本二"
    }
  ],
  "merge": true,
  "output_filename": "merged_output"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `items` | array | 是 | - | 配音列表，每项结构同 `/clone`（不含 `save_file`/`filename`） |
| `merge` | bool | 否 | true | 是否合并所有音频为一个文件 |
| `output_filename` | string | 否 | null | 合并后文件名（不含扩展名） |
| `return_raw` | bool | 否 | false | `true`=直接返回合并后 WAV 二进制流 |

**Response (200, return_raw=false):**
```json
{
  "success": true,
  "total": 2,
  "generated": 2,
  "files": [
    {
      "index": 0,
      "text": "第一段配音内容",
      "audio_url": "/output/batch_abc123_01.wav",
      "filename": "batch_abc123_01.wav"
    },
    {
      "index": 1,
      "text": "第二段配音内容",
      "audio_url": "/output/batch_abc123_02.wav",
      "filename": "batch_abc123_02.wav"
    }
  ],
  "merged": {
    "filename": "merged_output.wav",
    "audio_url": "/output/merged_output.wav"
  }
}
```
> `return_raw=true` 时返回合并后的 `audio/wav` 二进制流。

## 4. 对话生成

生成多角色对话，自动添加静音间隔和交叉淡入淡出。

```
POST /dialogue
```

**Request Body:**
```json
{
  "items": [
    {
      "text": "你好，请问有什么可以帮你的？",
      "ref_audio": "./audio/role_a.wav",
      "ref_text": "客服问候语"
    },
    {
      "text": "我想查询一下我的订单。",
      "ref_audio": "./audio/role_b.wav",
      "ref_text": "客户咨询"
    }
  ],
  "output_filename": "service_dialogue"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `items` | array | 是 | - | 对话列表，每项结构同批量克隆项 |
| `output_filename` | string | 否 | `"dialogue"` | 输出文件名（不含扩展名） |
| `return_raw` | bool | 否 | false | `true`=直接返回合并后 WAV 二进制流 |

**Response (200, return_raw=false):**
```json
{
  "success": true,
  "total_items": 2,
  "generated": 2,
  "audio_url": "/output/service_dialogue.wav",
  "filename": "service_dialogue.wav"
}
```
> `return_raw=true` 时返回合并后的 `audio/wav` 二进制流。

---

## 5. 语音转文本 (STT)

使用 Whisper 模型将音频转换为文字。

```
POST /stt
```

**Request Body:**
```json
{
  "ref_audio": "./audio/speech.wav"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ref_audio` | string | 是 | 音频文件路径 |

**Response (200):**
```json
{
  "success": true,
  "ref_audio": "./audio/speech.wav",
  "text": "识别出的文本内容"
}
```

---

## 6. VoxCPM2 生成

使用 VoxCPM2 扩散模型生成语音（需先 `POST /model/load {"model": "voxcpm2"}` 加载模型）。

### 6.1 声音克隆 + 情感

```
POST /vox/clone
```

**Request Body:**
```json
{
  "text": "这是一个使用 VoxCPM2 模型生成的语音",
  "ref_audio": "./audio/reference.wav",
  "ref_text": "参考音频文本",
  "instruct": "温暖的声音",
  "inference_timesteps": 5,
  "cfg_value": 3.0
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 目标文本，1-5000 字符 |
| `ref_audio` | string | 是 | - | 参考音频文件路径 |
| `ref_text` | string | 否 | null | 参考音频文本，1-5000 字符 |
| `instruct` | string | 否 | null | 情感描述，1-500 字符 |
| `inference_timesteps` | int | 否 | 5 | 扩散步数，1-10 |
| `cfg_value` | number | 否 | 3.0 | CFG 强度，0.5-5.0 |
| `save_file` | bool | 否 | true | `false`=直接返回 WAV 二进制流 |

**Response (200, save_file=true):**
```json
{
  "success": true,
  "text": "文本内容",
  "ref_audio": "./audio/reference.wav",
  "audio_url": "/output/vox_abc123.wav",
  "filename": "vox_abc123.wav",
  "processing_time": 2.34,
  "real_time_factor": 0.45,
  "audio_duration": 5.2
}
```
> `save_file=false` 时返回 `audio/wav` 二进制流。

### 6.2 声音设计

```
POST /vox/design
```

**Request Body:**
```json
{
  "text": "欢迎收听今天的新闻播报",
  "instruct": "沉稳的男声，语速适中，带有一点电台播音员的质感",
  "inference_timesteps": 7,
  "cfg_value": 3.0
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 目标文本，1-5000 字符 |
| `instruct` | string | 是 | - | 声音描述，1-500 字符 |
| `inference_timesteps` | int | 否 | 7 | 扩散步数，1-10 |
| `cfg_value` | number | 否 | 3.0 | CFG 强度，0.5-5.0 |
| `save_file` | bool | 否 | true | `false`=直接返回 WAV 二进制流 |

**Response (200, save_file=true):**
```json
{
  "success": true,
  "text": "欢迎收听今天的新闻播报",
  "instruct": "沉稳的男声，语速适中...",
  "audio_url": "/output/vox_def456.wav",
  "filename": "vox_def456.wav",
  "processing_time": 3.12,
  "real_time_factor": 0.52,
  "audio_duration": 6.0
}
```
> `save_file=false` 时返回 `audio/wav` 二进制流。

---

## 7. 文件管理

### 7.1 获取音频文件

```
GET /files/{filename}
```

返回 `audio/wav` 类型的文件响应。

> 文件名不能包含 `/`、`\` 或以 `.` 开头。

**Response (200):** 二进制 WAV 音频数据

### 7.2 列出生成文件

```
GET /files?limit=100&offset=0
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | int | 否 | 100 | 每页数量，1-1000 |
| `offset` | int | 否 | 0 | 偏移量 |

**Response (200):**
```json
{
  "files": [
    {
      "filename": "clone_abc12345.wav",
      "size": 123456,
      "url": "/output/clone_abc12345.wav"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### 7.3 直接静态访问

所有生成的音频文件也可通过 `/output/{filename}` 直接访问。

---

## 8. 错误码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| 200 | OK | 请求成功 |
| 400 | Bad Request | 参数校验失败（如参考音频不存在、文本为空、列表为空） |
| 404 | Not Found | 请求的文件不存在 |
| 500 | Internal Server Error | 服务端处理异常（生成失败、模型推理错误） |
| 503 | Service Unavailable | 模型未加载，需先调用 `/model/load` |

**错误响应格式:**
```json
{
  "detail": "错误描述信息"
}
```

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `TTS_SERVE_PORT` | `8000` | 服务器端口 |
| `TTS_SERVE_HOST` | `127.0.0.1` | 绑定地址 |
| `TTS_SERVE_LOG_LEVEL` | `warning` | 日志级别 |
| `TTS_SERVE_MODELS_DIR` | `./models` | 模型目录基础路径 |
| `TTS_SERVE_API_URL` | `http://localhost:8000` | WebUI 连接 API 的 URL |
| `TTS_SERVE_AUTO_START_API` | `1` | WebUI 是否自动启动 API 服务器 |

---

## 使用流程示例

```bash
# 1. 启动服务
python server_main.py

# 2. 加载 TTS 模型
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "tts"}'

# 3. 语音克隆
curl -X POST http://localhost:8000/clone \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "ref_audio": "./audio/sample.wav"}'

# 4. 获取音频
curl -O http://localhost:8000/output/clone_abc12345.wav

# 5. 语音转文本
curl -X POST http://localhost:8000/stt \
  -H "Content-Type: application/json" \
  -d '{"ref_audio": "./audio/speech.wav"}'
```
