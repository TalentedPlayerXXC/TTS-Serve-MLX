# 基于 FastAPI 的 TTS-Serve API
# 提供文本转语音、语音克隆、批量配音、语音转文本功能

import os
import uuid
import asyncio
import threading
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np

from tts_clone import TTSClone, merge_audio_list, save_audio, audio_to_wav_bytes, _merge_audio_arrays

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# 模型路径
MODELS_DIR = Path(os.environ.get("TTS_SERVE_MODELS_DIR", "./models"))
TTS_MODEL_PATH = str(MODELS_DIR / "qwenTTS_0.6B_MLX")
VOX_MODEL_PATH = str(MODELS_DIR / "voxCPM2_4bit_MLX")

OUTPUT_DIR = Path("./api_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载锁（防止并发重复加载）
_model_lock = threading.Lock()

_qwen_tts: Optional[TTSClone] = None
_vox = None


def _validate_audio_path(audio_path: str) -> Path:
    """校验音频路径安全性，防止路径遍历"""
    p = Path(audio_path).resolve()
    raw = Path(audio_path)
    if '..' in raw.parts:
        cwd = Path.cwd().resolve()
        if not str(p).startswith(str(cwd) + os.sep):
            raise HTTPException(status_code=400, detail="拒绝访问受保护路径之外的音频文件")
    return p


def _validate_filename(filename: str) -> str:
    """校验输出文件名安全性，防止路径遍历"""
    if '/' in filename or '\\' in filename or filename.startswith('.'):
        raise HTTPException(status_code=400, detail="无效的文件名")
    return filename


def load_qwen3():
    global _qwen_tts
    with _model_lock:
        if _qwen_tts is not None:
            logger.info("Qwen3 TTS 模型已加载，跳过")
            return

        logger.info("加载 Qwen3 TTS 模型: %s", TTS_MODEL_PATH)
        instance = TTSClone(model_path=TTS_MODEL_PATH)
        _ = instance.model  # 触发热加载，失败时抛异常，_qwen_tts 不会被错误赋值
        _qwen_tts = instance
        logger.info("Qwen3 TTS 模型加载完成")


def unload_qwen3():
    global _qwen_tts
    with _model_lock:
        if _qwen_tts:
            _qwen_tts.unload()
            _qwen_tts = None
        import gc
        gc.collect()
        import mlx.core as mx
        mx.set_cache_limit(0)
        mx.clear_cache()
        logger.info("Qwen3 TTS 模型已卸载，GPU 内存已释放")


def load_vox():
    global _vox
    with _model_lock:
        if _vox is not None:
            logger.info("VoxCPM2 模型已加载，跳过")
            return
        logger.info("加载 VoxCPM2 模型: %s", VOX_MODEL_PATH)
        from mlx_audio.tts.utils import load_model
        instance = load_model(VOX_MODEL_PATH)
        _vox = instance
        logger.info("VoxCPM2 模型加载完成")


def unload_vox():
    global _vox
    with _model_lock:
        _vox = None
        import gc
        gc.collect()
        import mlx.core as mx
        mx.set_cache_limit(0)
        mx.clear_cache()
        logger.info("VoxCPM2 模型已卸载，GPU 内存已释放")


# ============================================================
# FastAPI 依赖注入 — 模型就绪检查
# ============================================================

def require_qwen3():
    if _qwen_tts is None:
        raise HTTPException(status_code=503, detail="TTS 模型未加载，请先调用 POST /model/load {\"model\": \"tts\"}")


def require_vox():
    if _vox is None:
        raise HTTPException(status_code=503, detail="VoxCPM2 模型未加载，请先调用 POST /model/load {\"model\": \"voxcpm2\"}")


# ============================================================
# 启动和关闭事件
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TTS-Serve API 启动（无预载模型，请通过 /model/load 加载）")
    yield
    logger.info("正在释放模型资源...")
    unload_qwen3()
    unload_vox()
    logger.info("资源已释放")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="TTS-Serve API",
    description="""
## TTS-Serve 多功能 API

### 功能列表
- **TTS**：文本转语音（基础）
- **语音克隆**：使用参考音频克隆音色
- **批量配音**：批量生成多段配音

### 使用说明
1. 语音克隆提供 `text`（目标文本）和 `ref_audio`（参考音频路径）即可
2. 批量配音支持一次提交多段配音，可选合并
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（用于访问生成的音频）
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# ============================================================
# 请求/响应模型
# ============================================================

class CloneRequest(BaseModel):
    """语音克隆请求"""
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio: str = Field(..., min_length=1)
    ref_text: Optional[str] = Field(None, min_length=1, max_length=5000)
    stream: bool = False
    save_file: bool = True
    filename: Optional[str] = None


class BatchCloneItem(BaseModel):
    """批量克隆项目"""
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio: str = Field(..., min_length=1)
    ref_text: Optional[str] = Field(None, min_length=1, max_length=5000)
    stream: bool = False


class BatchCloneRequest(BaseModel):
    """批量配音请求"""
    items: List[BatchCloneItem]       # 配音列表
    merge: bool = True                # 是否合并所有音频
    output_filename: Optional[str] = None  # 合并后的文件名
    return_raw: bool = False          # True=直接返回合并后 WAV 流


class DialogueRequest(BaseModel):
    """对话场景请求"""
    items: List[BatchCloneItem]       # 对话列表
    output_filename: str = "dialogue" # 输出文件名
    return_raw: bool = False          # True=直接返回合并后 WAV 流
    silence_duration: float = 0.3     # 段落间静音间隔（秒）


class ModelLoadRequest(BaseModel):
    model: str = Field(..., pattern="^(tts|voxcpm2)$")


class ModelUnloadRequest(BaseModel):
    model: Optional[str] = Field(None, pattern="^(tts|voxcpm2)$")


class CleanupRequest(BaseModel):
    """缓存清理请求"""
    mode: str = Field(..., pattern="^(all|older_than|by_size)$")
    expire_hours: Optional[float] = Field(None, ge=0.1, le=720)
    max_size_mb: Optional[float] = Field(None, ge=1, le=100000)


class ModelDownloadRequest(BaseModel):
    """模型下载请求"""
    model: str = Field(..., pattern="^(qwenTTS_0.6B_MLX|voxCPM2_4bit_MLX)$")
    source: str = Field("modelscope", pattern="^(modelscope|huggingface)$")


class VoxCloneRequest(BaseModel):
    """VoxCPM2 声音克隆请求"""
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio: str = Field(..., min_length=1)
    ref_text: Optional[str] = Field(None, min_length=1, max_length=5000)
    instruct: Optional[str] = Field(None, min_length=1, max_length=500)
    inference_timesteps: int = Field(6, ge=1, le=10)
    cfg_value: float = Field(4.0, ge=0.5, le=5.0)
    save_file: bool = True            # False=直接返回 WAV 流


class VoxDesignRequest(BaseModel):
    """VoxCPM2 声音设计请求"""
    text: str = Field(..., min_length=1, max_length=5000)
    instruct: str = Field(..., min_length=1, max_length=500)
    inference_timesteps: int = Field(6, ge=1, le=10)
    cfg_value: float = Field(4.0, ge=0.5, le=5.0)
    save_file: bool = True            # False=直接返回 WAV 流


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "qwen3_loaded": _qwen_tts is not None,
        "voxcpm2_loaded": _vox is not None,
    }


@app.get("/model-info")
async def model_info():
    return {
        "qwen3": {
            "path": TTS_MODEL_PATH,
            "loaded": _qwen_tts is not None,
        },
        "voxcpm2": {
            "path": VOX_MODEL_PATH,
            "loaded": _vox is not None,
        },
    }


@app.get("/model/status")
async def model_status():
    return {
        "qwen3": _qwen_tts is not None,
        "voxcpm2": _vox is not None,
    }


@app.post("/model/load")
async def model_load(request: ModelLoadRequest):
    try:
        if request.model == "tts":
            load_qwen3()
        elif request.model == "voxcpm2":
            load_vox()
        return {"success": True, "model": request.model, "action": "loaded"}
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {e}")


@app.post("/model/unload")
async def model_unload(request: ModelUnloadRequest = None):
    if request is None or request.model is None:
        # 不传参数 = 全部卸载
        unload_qwen3()
        unload_vox()
        return {"success": True, "model": "all", "action": "unloaded"}

    if request.model == "tts":
        unload_qwen3()
    elif request.model == "voxcpm2":
        unload_vox()
    return {"success": True, "model": request.model, "action": "unloaded"}


# ============================================================
# 模型下载信息（供 Electron 前端下载用）
# ============================================================

_MODEL_SOURCES = {
    "qwenTTS_0.6B_MLX": {
        "name": "Qwen3-TTS 0.6B (4-bit)",
        "model_id": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        "size_gb": 1.6,
        "files": [
            "model.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "configuration.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/preprocessor_config.json",
        ],
        "sources": {
            "huggingface": "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
            "modelscope": "https://modelscope.cn/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        },
    },
    "voxCPM2_4bit_MLX": {
        "name": "VoxCPM2 2B (4-bit)",
        "model_id": "mlx-community/VoxCPM2-4bit",
        "size_gb": 2.1,
        "files": [
            "model.safetensors",
            "config.json",
            "configuration.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "sources": {
            "huggingface": "https://huggingface.co/mlx-community/VoxCPM2-4bit",
            "modelscope": "https://modelscope.cn/models/mlx-community/VoxCPM2-4bit",
        },
    },
}


@app.get("/models-info")
async def models_info():
    """返回模型下载信息和状态"""
    result = {}
    for key, info in _MODEL_SOURCES.items():
        model_dir = MODELS_DIR / key
        downloaded = model_dir.exists() and any(
            f.suffix == ".safetensors" for f in model_dir.iterdir()
        )
        result[key] = {
            "name": info["name"],
            "downloaded": downloaded,
            "size_gb": info["size_gb"],
            "files": info["files"],
            "sources": info["sources"],
        }
    return result


# ============================================================
# 模型下载
# ============================================================

_download_tasks: Dict[str, dict] = {}  # model_key → {status, progress, message}


async def _run_download(model_key: str, source: str):
    """后台下载模型（在独立线程中运行）"""
    info = _MODEL_SOURCES.get(model_key)
    if not info:
        _download_tasks[model_key] = {"status": "error", "progress": 0, "message": "未知模型"}
        return

    target_dir = MODELS_DIR / model_key
    target_dir.mkdir(parents=True, exist_ok=True)

    _download_tasks[model_key] = {"status": "downloading", "progress": 0, "message": "准备下载..."}

    try:
        import subprocess, re, threading

        if source == "modelscope":
            cmd = ["modelscope", "download", "--model", info["model_id"],
                   "--local_dir", str(target_dir)]
        else:
            cmd = ["huggingface-cli", "download", info["model_id"],
                   "--local-dir", str(target_dir)]

        def _do_download():
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # 从 stderr 解析进度（modelscope/hf-cli 的进度条输出在 stderr）
            pattern = re.compile(r"(\d+)%")
            for line in proc.stderr:
                line = line.strip()
                m = pattern.search(line)
                if m:
                    _download_tasks[model_key] = {
                        "status": "downloading",
                        "progress": int(m.group(1)),
                        "message": line[:80],
                    }
            proc.wait()
            if proc.returncode == 0:
                _download_tasks[model_key] = {"status": "completed", "progress": 100, "message": "下载完成"}
            else:
                stderr = proc.stderr.read() if proc.stderr else ""
                _download_tasks[model_key] = {"status": "error", "progress": 0, "message": stderr[:200]}
            # 保留 30 秒给前端轮询确认，之后自动清理
            threading.Timer(30, lambda: _download_tasks.pop(model_key, None)).start()

        thread = threading.Thread(target=_do_download, daemon=True)
        thread.start()

    except Exception as e:
        _download_tasks[model_key] = {"status": "error", "progress": 0, "message": str(e)}


@app.post("/model/download")
async def model_download(request: ModelDownloadRequest):
    """下载模型（异步启动，通过 GET /model/download/status/{model} 查进度）"""
    info = _MODEL_SOURCES.get(request.model)
    if info is None:
        raise HTTPException(status_code=400, detail=f"未知模型: {request.model}")

    target_dir = MODELS_DIR / request.model
    if target_dir.exists() and any(f.suffix == ".safetensors" for f in target_dir.iterdir()):
        return {"success": True, "model": request.model, "action": "already_downloaded"}

    # 检查是否已在下载
    if request.model in _download_tasks:
        task = _download_tasks[request.model]
        if task["status"] == "downloading":
            return {"success": True, "model": request.model, "action": "already_downloading"}

    await _run_download(request.model, request.source)
    return {"success": True, "model": request.model, "action": "started", "status_url": f"/model/download/status/{request.model}"}


@app.get("/model/download/status/{model_key}")
async def model_download_status(model_key: str):
    """查询模型下载进度"""
    task = _download_tasks.get(model_key)
    if not task:
        # 检查是否已存在
        target_dir = MODELS_DIR / model_key
        exists = target_dir.exists() and any(f.suffix == ".safetensors" for f in target_dir.iterdir())
        return {"model": model_key, "status": "completed" if exists else "not_started", "progress": 100 if exists else 0}
    return {"model": model_key, **task}


# ============================================================
# 语音克隆接口
# ============================================================

@app.post("/clone")
async def voice_clone(request: CloneRequest):
    """
    语音克隆接口
    
    使用参考音频的音色和情感风格，生成新的语音
    
    - **text**: 要转换的目标文本
    - **ref_audio**: 参考音频文件路径（相对于工作目录）
    - **ref_text**: 参考音频对应的原始文本
    - **save_file**: 是否保存为文件（默认 True）
    - **filename**: 自定义文件名（不含扩展名）
    
    Returns:
        - **audio_url**: 生成的音频文件访问路径（如果 save_file=True）
        - **audio_data**: base64 编码的音频数据（可选返回）
    """
    require_qwen3()
    
    # 验证参考音频
    ref_path = _validate_audio_path(request.ref_audio)
    if not ref_path.exists():
        raise HTTPException(status_code=400, detail=f"参考音频文件不存在: {request.ref_audio}")
    
    # 生成文件名（仅 save_file=True 时保存）
    save_file = request.save_file
    if save_file:
        filename = request.filename or None
        if filename:
            filename = f"{filename}.wav"
        else:
            filename = f"clone_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_DIR / filename
    else:
        output_path = None
        filename = None
    
    try:
        # 调用 TTS 克隆
        audio = _qwen_tts.generate(
            text=request.text,
            ref_audio=str(ref_path),
            ref_text=request.ref_text,
            stream=request.stream,
            output_path=str(output_path) if output_path else None,
        )
        
        if audio is None:
            raise HTTPException(
                status_code=500,
                detail=f"音频生成失败（ref_audio={request.ref_audio}，请检查参考音频是否可访问）"
            )
        
        if not save_file:
            return Response(content=audio_to_wav_bytes(audio, 24000), media_type="audio/wav")

        result = {
            "success": True,
            "text": request.text,
            "ref_audio": request.ref_audio,
            "audio_url": f"/output/{filename}",
            "filename": filename,
        }
        return result
        
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# ============================================================
# 批量配音接口
# ============================================================

@app.post("/batch-clone")
async def batch_clone(request: BatchCloneRequest):
    """
    批量配音接口

    一次提交多段配音，自动批量生成并可选合并

    - **items**: 配音列表，每项包含 text、ref_audio
    - **merge**: 是否合并所有音频为一个文件
    - **output_filename**: 合并后的文件名
    """
    require_qwen3()

    if not request.items:
        raise HTTPException(status_code=400, detail="配音列表不能为空")

    # ---- 预校验所有路径 ----
    valid_items = []
    for i, item in enumerate(request.items):
        try:
            ref_path = _validate_audio_path(item.ref_audio)
        except HTTPException:
            logger.warning("第 %d 项参考音频路径不合法，跳过: %s", i + 1, item.ref_audio)
            continue
        if not ref_path.exists():
            logger.warning("第 %d 项参考音频不存在，跳过: %s", i + 1, item.ref_audio)
            continue
        valid_items.append((i, item, ref_path))

    if not valid_items:
        raise HTTPException(status_code=400, detail="所有配音项的参考音频均无效")

    results = []
    audio_list = []
    audio_cache = {}  # ref_audio路径 → mx.array（避免重复读盘）

    try:
        # ---- 判断是否能走模型原生 batch ----
        # 条件：不流式 + 所有 items 用同一个 ref_audio
        all_same_ref = len(set(str(rp) for _, _, rp in valid_items)) == 1

        if all_same_ref and not any(item.stream for _, item, _ in valid_items)\
           and all(item.ref_text for _, item, _ in valid_items):
            # 走模型原生 batch_generate（仅 ICL 模式支持，Speaker 模式不走这里）
            _, _, ref_path = valid_items[0]
            logger.info("批量配音: %d 段共用参考音频，走模型原生 batch", len(valid_items))

            texts = [item.text for _, item, _ in valid_items]
            batch_results = list(_qwen_tts.model.batch_generate(
                texts=texts,
                ref_audio=str(ref_path),
                ref_text=None,
                stream=False,
            ))

            for idx, (i, item, _) in enumerate(valid_items):
                if idx < len(batch_results):
                    audio = np.array(batch_results[idx].audio)
                    audio_list.append(audio)

                    _qwen_tts._save_audio_if_needed(audio, OUTPUT_DIR, i)
                    results.append({
                        "index": i,
                        "text": item.text,
                        "sample_rate": 24000,
                    })
        else:
            # ---- 逐条生成（带音频读取缓存 + 按角色复用）----
            logger.info("批量配音: %d 段逐条生成", len(valid_items))

            for i, item, ref_path in valid_items:
                ref_key = str(ref_path)

                audio = _qwen_tts.generate(
                    text=item.text,
                    ref_audio=ref_key,
                    ref_text=item.ref_text,
                    stream=item.stream,
                    output_path=None,  # 不写磁盘
                )

                if audio is not None:
                    audio_list.append(audio)
                    results.append({
                        "index": i,
                        "text": item.text,
                        "sample_rate": 24000,
                    })
        
        response = {
            "success": True,
            "total": len(request.items),
            "generated": len(results),
        }

        # 合并音频
        if request.merge and audio_list:
            if request.return_raw:
                merged = _merge_audio_arrays(audio_list, 24000)
                if merged is not None:
                    return Response(content=audio_to_wav_bytes(merged, 24000), media_type="audio/wav")

            merged_filename = request.output_filename or f"merged_{uuid.uuid4().hex[:8]}"
            merged_path = OUTPUT_DIR / f"{merged_filename}.wav"
            merge_audio_list(audio_list, merged_path, verbose=False)
            response["merged"] = {
                "filename": f"{merged_filename}.wav",
                "audio_url": f"/output/{merged_filename}.wav",
            }
            response["files"] = results
        else:
            # 非 merge 模式：保存单个文件并返回路径
            for i, item_info in enumerate(results):
                if i < len(audio_list):
                    fname = f"batch_{uuid.uuid4().hex[:8]}_{i+1:02d}.wav"
                    fpath = OUTPUT_DIR / fname
                    save_audio(audio_list[i], fpath, verbose=False)
                    results[i]["audio_url"] = f"/output/{fname}"
                    results[i]["filename"] = fname
            response["files"] = results
        
        return response
        
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


# ============================================================
# 对话场景接口
# ============================================================

@app.post("/dialogue")
async def generate_dialogue(request: DialogueRequest):
    """
    对话场景接口

    生成多角色对话，自动添加静音间隔和交叉淡入淡出

    - **items**: 对话列表
    - **output_filename**: 输出文件名（不含扩展名）
    - **silence_duration**: 段落间静音间隔秒数（默认0.3）
    """
    require_qwen3()

    if not request.items:
        raise HTTPException(status_code=400, detail="对话列表不能为空")

    # ---- 预校验所有路径 ----
    valid_items = []
    for i, item in enumerate(request.items):
        try:
            ref_path = _validate_audio_path(item.ref_audio)
        except HTTPException:
            logger.warning("第 %d 项参考音频路径不合法，跳过: %s", i + 1, item.ref_audio)
            continue
        if not ref_path.exists():
            logger.warning("第 %d 项参考音频不存在，跳过: %s", i + 1, item.ref_audio)
            continue
        valid_items.append((i, item, ref_path))

    if not valid_items:
        raise HTTPException(status_code=400, detail="所有对话项的参考音频均无效")

    audio_list = []

    try:
        for i, item, ref_path in valid_items:
            audio = _qwen_tts.generate(
                text=item.text,
                ref_audio=str(ref_path),
                ref_text=item.ref_text,
                stream=item.stream,
                output_path=None,
            )
            if audio is not None:
                audio_list.append(audio)

        if not audio_list:
            raise HTTPException(status_code=500, detail="所有配音生成失败")

        if request.return_raw:
            merged = _merge_audio_arrays(audio_list, 24000, request.silence_duration)
            if merged is not None:
                return Response(content=audio_to_wav_bytes(merged, 24000), media_type="audio/wav")

        # 合并保存
        output_path = OUTPUT_DIR / f"{request.output_filename}.wav"
        merge_audio_list(audio_list, output_path, silence_duration=request.silence_duration, verbose=False)
        
        return {
            "success": True,
            "total_items": len(request.items),
            "generated": len(audio_list),
            "audio_url": f"/output/{request.output_filename}.wav",
            "filename": f"{request.output_filename}.wav",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")


# ============================================================
# VoxCPM2 生成接口
# ============================================================

def _vox_generate_and_save(kwargs: dict) -> dict:
    """执行 VoxCPM2 推理并保存音频，返回响应 JSON 字段"""
    results = list(_vox.generate(**kwargs))
    if not results:
        raise HTTPException(status_code=500, detail="VoxCPM2 生成失败")

    audio = np.array(results[0].audio)
    output_path = OUTPUT_DIR / f"vox_{uuid.uuid4().hex[:8]}.wav"
    save_audio(audio, output_path, sample_rate=48000, verbose=False)

    return {
        "audio_url": f"/output/{output_path.name}",
        "filename": output_path.name,
        "processing_time": results[0].processing_time_seconds,
        "real_time_factor": results[0].real_time_factor,
        "audio_duration": results[0].audio_duration,
        "_raw_audio": audio,
    }


def _vox_generate_raw(kwargs: dict) -> np.ndarray:
    """执行 VoxCPM2 推理，仅返回 numpy 音频数组"""
    results = list(_vox.generate(**kwargs))
    if not results:
        raise HTTPException(status_code=500, detail="VoxCPM2 生成失败")
    return np.array(results[0].audio)


@app.post("/vox/clone")
async def vox_clone(request: VoxCloneRequest):
    require_vox()
    ref_path = _validate_audio_path(request.ref_audio)
    if not ref_path.exists():
        raise HTTPException(status_code=400, detail=f"参考音频文件不存在: {request.ref_audio}")

    try:
        kwargs = {
            "text": request.text,
            "ref_audio": str(ref_path),
            "inference_timesteps": request.inference_timesteps,
            "cfg_value": request.cfg_value,
        }
        if request.ref_text:
            kwargs["ref_text"] = request.ref_text
        if request.instruct:
            kwargs["instruct"] = request.instruct

        if not request.save_file:
            audio = _vox_generate_raw(kwargs)
            return Response(content=audio_to_wav_bytes(audio, 48000), media_type="audio/wav")

        result = _vox_generate_and_save(kwargs)
        result.pop("_raw_audio", None)
        result["success"] = True
        result["text"] = request.text
        result["ref_audio"] = request.ref_audio
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"VoxCPM2 克隆失败: {str(e)}")


@app.post("/vox/design")
async def vox_design(request: VoxDesignRequest):
    require_vox()

    try:
        kwargs = {
            "text": request.text,
            "instruct": request.instruct,
            "inference_timesteps": request.inference_timesteps,
            "cfg_value": request.cfg_value,
        }

        if not request.save_file:
            audio = _vox_generate_raw(kwargs)
            return Response(content=audio_to_wav_bytes(audio, 48000), media_type="audio/wav")

        result = _vox_generate_and_save(kwargs)
        result.pop("_raw_audio", None)
        result["success"] = True
        result["text"] = request.text
        result["instruct"] = request.instruct
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"VoxCPM2 声音设计失败: {str(e)}")


# ============================================================
# 获取输出文件
# ============================================================

@app.get("/files/{filename}")
async def get_file(filename: str):
    """获取生成的文件"""
    safe_name = _validate_filename(filename)
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.get("/files")
async def list_files(limit: int = Query(default=100, ge=1, le=1000),
                      offset: int = Query(default=0, ge=0)):
    """列出所有生成的文件（支持分页）"""
    all_files = sorted(
        [f for f in OUTPUT_DIR.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    total = len(all_files)
    page = all_files[offset:offset + limit]
    files = []
    for f in page:
        files.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "url": f"/output/{f.name}",
        })
    return {"files": files, "total": total, "limit": limit, "offset": offset}


# ============================================================
# 缓存信息
# ============================================================

@app.get("/cache")
async def cache_info():
    """查看缓存状态：文件数、总大小、最旧/最新文件"""
    all_files = sorted(
        [f for f in OUTPUT_DIR.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
    )
    total_files = len(all_files)
    total_bytes = sum(f.stat().st_size for f in all_files)
    oldest = all_files[0].name if all_files else None
    newest = all_files[-1].name if all_files else None
    oldest_age = time.time() - all_files[0].stat().st_mtime if all_files else 0

    return {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "oldest_file": oldest,
        "newest_file": newest,
        "oldest_age_hours": round(oldest_age / 3600, 1),
    }


# ============================================================
# 缓存清理
# ============================================================

@app.post("/cleanup")
async def cleanup_cache(request: CleanupRequest):
    """清理生成的音频缓存

    三种模式:
    - all: 一键清空
    - older_than: 按过期时间（小时），如 {"mode": "older_than", "expire_hours": 24}
    - by_size: 按占用大小（MB）保留最新的，如 {"mode": "by_size", "max_size_mb": 500}
    """
    all_files = sorted(
        [f for f in OUTPUT_DIR.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    now = time.time()
    deleted = []
    kept = []

    if request.mode == "all":
        # 全部删除
        for f in all_files:
            size = f.stat().st_size
            f.unlink()
            deleted.append({"filename": f.name, "size": size})
        logger.info("清理全部缓存: %d 个文件", len(deleted))

    elif request.mode == "older_than":
        if request.expire_hours is None:
            raise HTTPException(status_code=400, detail="older_than 模式需要 expire_hours 参数")
        cutoff = now - request.expire_hours * 3600
        for f in all_files:
            if f.stat().st_mtime < cutoff:
                size = f.stat().st_size
                f.unlink()
                deleted.append({"filename": f.name, "size": size})
            else:
                kept.append(f.name)
        logger.info("清理 %d 小时前的缓存: 删 %d 个, 留 %d 个",
                     request.expire_hours, len(deleted), len(kept))

    elif request.mode == "by_size":
        if request.max_size_mb is None:
            raise HTTPException(status_code=400, detail="by_size 模式需要 max_size_mb 参数")
        max_bytes = request.max_size_mb * 1024 * 1024
        total_bytes = sum(f.stat().st_size for f in all_files)

        if total_bytes <= max_bytes:
            kept = [f.name for f in all_files]
            logger.info("缓存大小 %.1fMB 未超限 %.1fMB，无需清理",
                         total_bytes / 1024 / 1024, request.max_size_mb)
        else:
            # 从最旧的开始删，直到低于上限
            # all_files 已按 mtime 降序（最新在前），反过来从最后删
            to_delete = []
            to_keep = []
            running_total = 0
            for f in reversed(all_files):  # 从最旧开始
                size = f.stat().st_size
                if running_total + size > max_bytes:
                    to_delete.append((f, size))
                else:
                    running_total += size
                    to_keep.append(f)

            for f, size in to_delete:
                f.unlink()
                deleted.append({"filename": f.name, "size": size})
            kept = [f.name for f in to_keep]
            logger.info("按大小清理: 上限 %.1fMB, 当前 %.1fMB, 删 %d 个, 留 %d 个",
                         request.max_size_mb, total_bytes / 1024 / 1024,
                         len(deleted), len(kept))

    total_deleted = sum(d["size"] for d in deleted)
    return {
        "success": True,
        "mode": request.mode,
        "deleted_count": len(deleted),
        "kept_count": len(kept),
        "freed_bytes": total_deleted,
        "freed_mb": round(total_deleted / 1024 / 1024, 2),
        "deleted_files": [d["filename"] for d in deleted[:20]],  # 最多列20个
        "kept_files": kept[:20],
    }


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TTS_SERVE_PORT", "8000"))
    host = os.environ.get("TTS_SERVE_HOST", "127.0.0.1")
    log_level = os.environ.get("TTS_SERVE_LOG_LEVEL", "warning")

    logger.info("=" * 60)
    logger.info("启动 TTS-Serve API 服务（启动时不预载模型）")
    logger.info("=" * 60)
    logger.info("Qwen3 TTS（Speaker 模式）: %s", TTS_MODEL_PATH)
    logger.info("VoxCPM2（情感克隆/设计）: %s", VOX_MODEL_PATH)
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("API 文档: http://%s:%d/docs", host, port)
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level=log_level)
