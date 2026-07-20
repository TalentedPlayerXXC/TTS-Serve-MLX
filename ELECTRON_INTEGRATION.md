# TTS-Serve-MLX Electron 集成指南

## 架构概览

```
Electron 主进程
  │   spawn / child_process
  │
  └─→ Python Server (子进程, localhost:8000)
       │
       ├── GET  /health              → 健康检查
       ├── GET  /model-info          → 模型状态详情
       ├── GET  /model/status        → 模型加载状态
       ├── POST /model/load          → 加载模型 (tts / voxcpm2)
       ├── POST /model/unload        → 卸载模型
       │
       ├── POST /clone               → 语音克隆
       ├── POST /batch-clone         → 批量配音
       ├── POST /dialogue            → 多角色对话
       │
       ├── POST /vox/clone           → VoxCPM2 克隆+情感
       ├── POST /vox/design          → VoxCPM2 声音设计
       │
       ├── GET  /files               → 文件列表
       ├── GET  /files/{filename}    → 下载文件
       └── /output/{filename}        → 静态文件访问 (FastAPI mount)
```

---

## 安全说明：Origin/Referer 校验

> ⚠️ **后端已内置 Origin/Referer 校验中间件，Electron 不需要任何额外适配。**

后端对所有 POST/PUT/DELETE 请求检查 `Origin` / `Referer` 头：

| 来源 | 行为 | 原因 |
|------|------|------|
| 空（无 Origin 也无 Referer） | ✅ **放行** | curl / Python 脚本 / **Electron 主进程** |
| `Origin: null` | ✅ **放行** | Electron 渲染进程 `file://` 协议 |
| `localhost:xxxx` | ✅ **放行** | Swagger UI、Vite 开发服务器 |
| `127.0.0.1:xxxx` | ✅ **放行** | 同上 |
| 其他任意来源 | ❌ **403 拒绝** | 恶意网页 `<form>` / `fetch` 绕过 CORS 打后端 |

> **原理**：当用户在 Electron 中浏览恶意网站时，浏览器会自动在请求头中带上 `Origin: https://evil.com`，中间件看到这个来源不是本地，直接拒绝。恶意网站没办法伪造 Origin（浏览器安全策略），所以这个方案对「开着 Electron 访问恶意网页」的场景覆盖得很干净。

前端调用（主进程 `spawn` 出来的 Python 进程、渲染进程 `file://` 的 fetch）都不带非本地 Origin，所以完全不受影响，**零改动**。

---

## 1. 放置打包产物

将 `dist/tts_serve_mlx/` 复制到 Electron 项目的资源目录：

```
your-electron-app/
  └── resources/
      └── python-server/
          ├── tts_serve_mlx/      ← 打包产物 (不含模型)
          │   ├── tts_serve_mlx   ← 可执行文件
          │   └── _internal/
           └── models/             ← 由 Electron 下载，无需手动放置
               ├── qwenTTS_0.6B_MLX/
               └── voxCPM2_4bit_MLX/
```

> **模型不需要手动放置**。启动时 Electron 通过 HTTP 从魔搭下载，详细流程见「模型下载」章节。

## 2. 模型下载（推荐方案）

服务启动后，调 `POST /model/download` 一键下载模型，后端 HTTP 流式下载，前端轮询进度。

```typescript
/** 一键下载模型，带进度反馈 */
async function downloadModel(modelKey: string): Promise<boolean> {
  // 1. 发起下载
  const startRes = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/model/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelKey, source: 'modelscope' }),
  });
  const startData = await startRes.json();
  if (!startRes.ok || startData.action === 'already_downloaded') {
    return startRes.ok;
  }

  // 2. 轮询下载进度（建议前端加指数退避：1s → 2s → 4s，大型模型下载 2GB+ 可减少请求数）
  return new Promise((resolve) => {
    const interval = setInterval(async () => {
      const statusRes = await fetch(
        `http://${SERVER_HOST}:${SERVER_PORT}/model/download/status/${modelKey}`
      );
      const status = await statusRes.json();
      console.log(`[TTS] ${modelKey}: ${status.progress}% - ${status.message}`);

      if (status.status === 'completed') {
        clearInterval(interval);
        resolve(true);
      } else if (status.status === 'error') {
        clearInterval(interval);
        resolve(false);
      }
    }, 2000);
  });
}

// ---- 完整启动流程 ----
async function startApp() {
  await startTTSServer();

  const dl1 = downloadModel('qwenTTS_0.6B_MLX');
  // 显示 "Qwen3-TTS: 45% - Downloading [model.safetensors]: 45%|█████"

  const dl2 = downloadModel('voxCPM2_4bit_MLX');

  await Promise.all([dl1, dl2]);
  await loadModel('tts');
  createWindow();
}
```

> 魔搭源优先（`source: 'modelscope'`），国内无需代理。
> 下载完成后模型自动放在正确位置，直接 `POST /model/load` 即可使用。
> 已下载的模型会跳过，不会重复下载。

## 3. Electron 主进程代码 (main.ts)
```typescript
import { app, BrowserWindow } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as http from 'http';

let pythonServer: ChildProcess | null = null;
let serverPort: number = 0;  // 从 Python stdout 解析

// ---- 配置 ----
const SERVER_HOST = '127.0.0.1';
const STARTUP_TIMEOUT = 60_000; // 60 秒
const HEALTH_CHECK_INTERVAL = 500; // 500ms

// ---- 判断运行模式 ----
function getServerPath(): { exe: string; args: string[]; cwd: string; modelsDir?: string } {
    if (app.isPackaged) {
        // 生产模式: 使用打包好的 Python 可执行文件
        // 打包产物不含模型文件，需通过 TTS_SERVE_MODELS_DIR 指定模型路径
        const serverDir = path.join(process.resourcesPath, 'python-server', 'tts_serve_mlx');
        const modelsDir = path.join(process.resourcesPath, 'python-server', 'models');
        return {
            exe: path.join(serverDir, 'tts_serve_mlx'),
            args: [],
            cwd: serverDir,
            modelsDir,
        };
    } else {
        // 开发模式: 直接使用系统 Python
        // cwd 需指向 TTS-Serve-MLX 项目根目录（含 server_main.py）
        return {
            exe: 'python3',
            args: ['server_main.py'],
            cwd: path.join(__dirname, '..'),
        };
    }
}

// ---- 启动 Python TTS 服务 ----
async function startTTSServer(): Promise<void> {
    const { exe, args, cwd, modelsDir } = getServerPath();

    console.log(`[TTS] 启动服务: ${exe} ${args.join(' ')}`);

    pythonServer = spawn(exe, args, {
        cwd,
        env: {
            ...process.env,
            // 不传 TTS_SERVE_PORT，让 Python 自动找空闲端口（8000-8050）
            TTS_SERVE_HOST: SERVER_HOST,
            TTS_SERVE_LOG_LEVEL: 'warning',
            ...(modelsDir ? { TTS_SERVE_MODELS_DIR: modelsDir } : {}),
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    // 从 stdout 解析端口号
    pythonServer.stdout?.on('data', (data: Buffer) => {
        const text = data.toString().trim();
        console.log(`[TTS:stdout] ${text}`);
        const match = text.match(/^TTS_SERVER_PORT=(\d+)$/);
        if (match) {
            serverPort = parseInt(match[1], 10);
            console.log(`[TTS] 发现动态端口: ${serverPort}`);
        }
    });
    pythonServer.stderr?.on('data', (data: Buffer) => {
        console.log(`[TTS:stderr] ${data.toString().trim()}`);
    });
    pythonServer.on('exit', (code: number | null) => {
        console.log(`[TTS] 进程退出, code=${code}`);
        pythonServer = null;
    });
    pythonServer.on('error', (err: Error) => {
        console.error(`[TTS] 进程错误:`, err.message);
    });

    // 等待端口发现 + 服务就绪
    await waitForPortAndHealth();
    console.log('[TTS] 服务就绪!');
}

/** 等待 Python 打印端口号，然后等待健康检查 */
async function waitForPortAndHealth(): Promise<void> {
    const start = Date.now();
    while (Date.now() - start < STARTUP_TIMEOUT) {
        if (serverPort > 0) {
            const ok = await httpGet(`http://${SERVER_HOST}:${serverPort}/health`);
            if (ok) return;
        }
        await new Promise((r) => setTimeout(r, HEALTH_CHECK_INTERVAL));
    }
    throw new Error(`TTS 服务启动超时 (${STARTUP_TIMEOUT}ms)`);
}

> ⚠️ **后续 API 调用**：所有请求地址中的 `:8000` 应替换为 `:${serverPort}`，后端已不固定使用 8000 端口。

// ---- 健康检查轮询 ----
function httpGet(url: string): Promise<boolean> {
    return new Promise((resolve) => {
        const req = http.get(url, (res) => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(2000, () => { req.destroy(); resolve(false); });
    });
}

async function waitForHealth(): Promise<void> {
    const url = `http://${SERVER_HOST}:${SERVER_PORT}/health`;
    const start = Date.now();

    while (Date.now() - start < STARTUP_TIMEOUT) {
        const ok = await httpGet(url);
        if (ok) return;
        await new Promise((r) => setTimeout(r, HEALTH_CHECK_INTERVAL));
    }

    throw new Error(`TTS 服务启动超时 (${STARTUP_TIMEOUT}ms)`);
}

// ---- 关闭服务 ----
function stopTTSServer(): void {
    if (!pythonServer) return;

    console.log('[TTS] 正在关闭服务...');

    // 先 SIGTERM 优雅关闭
    pythonServer.kill('SIGTERM');

    // 5 秒后强制 SIGKILL
    setTimeout(() => {
        if (pythonServer) {
            console.log('[TTS] 强制终止');
            pythonServer.kill('SIGKILL');
        }
    }, 5000);
}

// ---- API 调用工具函数 ----

/** 语音克隆 */
async function ttsClone(
    text: string,
    refAudio: string,
    refText?: string,
    stream?: boolean,
    filename?: string,
    saveFile: boolean = true,
) {
    const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            ref_audio: refAudio,
            ref_text: refText ?? null,
            stream: stream ?? false,
            save_file: saveFile,
            filename: filename ?? null,
        }),
    });
    if (!saveFile) return res.arrayBuffer();
    return res.json();
}

/** 批量配音 */
async function ttsBatchClone(
    items: Array<{
        text: string;
        ref_audio: string;
        ref_text?: string;
        stream?: boolean;
    }>,
    merge: boolean = true,
    returnRaw: boolean = false,
) {
    const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/batch-clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items, merge, return_raw: returnRaw }),
    });
    if (returnRaw) return res.arrayBuffer();
    return res.json();
}

/** VoxCPM2 克隆 + 情感 */
async function voxClone(text: string, refAudio: string, options?: {
    refText?: string;
    instruct?: string;
    inferenceTimesteps?: number;
    cfgValue?: number;
    saveFile?: boolean;
}) {
    const saveFile = options?.saveFile ?? true;
    const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/vox/clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            ref_audio: refAudio,
            ref_text: options?.refText ?? null,
            instruct: options?.instruct ?? null,
            inference_timesteps: options?.inferenceTimesteps ?? 5,
            cfg_value: options?.cfgValue ?? 3.0,
            save_file: saveFile,
        }),
    });
    if (!saveFile) return res.arrayBuffer();
    return res.json();
}

/** VoxCPM2 声音设计 */
async function voxDesign(text: string, instruct: string, options?: {
    inferenceTimesteps?: number;
    cfgValue?: number;
    saveFile?: boolean;
}) {
    const saveFile = options?.saveFile ?? true;
    const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/vox/design`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            instruct,
            inference_timesteps: options?.inferenceTimesteps ?? 7,
            cfg_value: options?.cfgValue ?? 3.0,
            save_file: saveFile,
        }),
    });
    if (!saveFile) return res.arrayBuffer();
    return res.json();
}

/** 加载模型（服务启动后必须先加载模型才能使用） */
async function loadModel(model: 'tts' | 'voxcpm2'): Promise<boolean> {
    try {
        const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/model/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
        });
        return res.ok;
    } catch {
        return false;
    }
}

/** 卸载模型 */
async function unloadModel(model: 'tts' | 'voxcpm2'): Promise<boolean> {
    try {
        const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/model/unload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
        });
        return res.ok;
    } catch {
        return false;
    }
}

/** 获取生成的文件 URL */
function getAudioUrl(filename: string): string {
    return `http://${SERVER_HOST}:${SERVER_PORT}/output/${filename}`;
}

// ---- Electron 生命周期集成 ----

app.whenReady().then(async () => {
    try {
        await startTTSServer();

        // 加载 TTS 模型，服务就绪后必须加载才能使用
        console.log('[TTS] 正在加载 Qwen3 TTS 模型...');
        const loaded = await loadModel('tts');
        if (!loaded) {
            throw new Error('TTS 模型加载失败');
        }
        console.log('[TTS] 模型加载完成');
    } catch (err) {
        console.error('[TTS] 启动失败:', err);
        app.quit();
        return;
    }

    // 创建窗口
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
        },
    });

    mainWindow.loadURL('http://localhost:5173'); // 或你的页面地址

    // 通过 contextBridge 暴露 API 给渲染进程 (preload.js)
    // 略...
});

app.on('will-quit', () => {
    stopTTSServer();
});

app.on('window-all-closed', () => {
    stopTTSServer();
    if (process.platform !== 'darwin') app.quit();
});
```

## 3. 在 `electron-builder` 中配置

在 `electron-builder.yml` 中配置额外资源打包：

```yaml
# electron-builder.yml
extraResources:
  - from: resources/python-server/
    to: python-server/
    filter:
      - "**/*"
```

## 4. 开发模式运行

在开发时，不需要每次启动都打包，直接让 Electron 使用系统 Python：

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 启动 Python 服务
python3 server_main.py

# 在另一个终端启动 Electron
npm run dev
```

## 5. 常见问题

### 端口被占用
修改 `TTS_SERVE_PORT` 环境变量：
```typescript
env: { TTS_SERVE_PORT: '0' } // 0 = 随机端口
```
然后在 stdout 中解析实际端口。

### 多实例支持
如果需要多个 TTS 实例，每次使用不同端口。

### 缓存管理
Python 服务会积累生成的音频文件在 `api_output/` 目录中。可以通过 API 管理缓存：

```typescript
/** 查看缓存状态 */
async function getCacheStatus() {
  const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/cache`);
  return res.json();
  // { total_files: 42, total_mb: 50, oldest_age_hours: 72.5, ... }
}

/** 清理缓存（三种模式） */
async function cleanupCache(mode: 'all' | 'older_than' | 'by_size', value?: number) {
  const body: any = { mode };
  if (mode === 'older_than') body.expire_hours = value ?? 24;
  if (mode === 'by_size') body.max_size_mb = value ?? 500;
  const res = await fetch(`http://${SERVER_HOST}:${SERVER_PORT}/cleanup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}
```

### Python 日志
Python 服务通过 stdout/stderr 输出日志，可在 Electron 的 Console 中查看。
