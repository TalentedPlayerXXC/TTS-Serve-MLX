# 基于 FastAPI 的 TTS-Serve API
# 提供文本转语音、语音克隆、批量配音、语音转文本功能

import os
import uuid
import threading
import logging
from pathlib import Path
from typing import Optional, List
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
ASR_MODEL_PATH = str(MODELS_DIR / "whisper_asr_MLX")
VOX_MODEL_PATH = str(MODELS_DIR / "voxCPM2_4bit_MLX")

OUTPUT_DIR = Path("./api_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载锁（防止并发重复加载）
_model_lock = threading.Lock()

_qwen_tts: Optional[TTSClone] = None
_qwen_asr = None
_vox = None
_asr_loaded_at = None  # 记录 ASR 加载路径，用于判断是否需要重新注入


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


def _inject_asr_to_tts():
    global _qwen_tts, _qwen_asr, _asr_loaded_at
    if _qwen_tts is not None and _qwen_asr is not None:
        if _qwen_tts.asr_model is None or _asr_loaded_at != id(_qwen_asr):
            _qwen_tts.asr_model = _qwen_asr
            _asr_loaded_at = id(_qwen_asr)
            logger.info("ASR 模型已注入到 TTSClone")


def load_qwen3():
    global _qwen_tts, _qwen_asr
    with _model_lock:
        if _qwen_tts is not None and _qwen_asr is not None:
            logger.info("TTS 模型已加载，跳过")
            return

        if _qwen_tts is None:
            logger.info("加载 Qwen3 TTS 模型: %s", TTS_MODEL_PATH)
            _qwen_tts = TTSClone(model_path=TTS_MODEL_PATH)
            _ = _qwen_tts.model
            logger.info("Qwen3 TTS 模型加载完成")

        if _qwen_asr is None:
            logger.info("加载 Whisper ASR 模型: %s", ASR_MODEL_PATH)
            from mlx_audio.stt import load as load_whisper
            _qwen_asr = load_whisper(ASR_MODEL_PATH)
            logger.info("ASR 模型加载完成")
            _inject_asr_to_tts()


def unload_qwen3():
    global _qwen_tts, _qwen_asr
    with _model_lock:
        if _qwen_tts:
            _qwen_tts.asr_model = None
            _qwen_tts.unload()
            _qwen_tts = None
        _qwen_asr = None
        import gc
        gc.collect()
        logger.info("TTS 模型已卸载")


def load_vox():
    global _vox
    with _model_lock:
        if _vox is not None:
            logger.info("VoxCPM2 模型已加载，跳过")
            return
        logger.info("加载 VoxCPM2 模型: %s", VOX_MODEL_PATH)
        from mlx_audio.tts.utils import load_model
        _vox = load_model(VOX_MODEL_PATH)
        logger.info("VoxCPM2 模型加载完成")


def unload_vox():
    global _vox
    with _model_lock:
        _vox = None
        import gc
        gc.collect()
        logger.info("VoxCPM2 模型已卸载")


# ============================================================
# FastAPI 依赖注入 — 模型就绪检查
# ============================================================

def require_qwen3():
    if _qwen_tts is None:
        raise HTTPException(status_code=503, detail="TTS 模型未加载，请先调用 POST /model/load {\"model\": \"tts\"}")


def require_asr():
    if _qwen_asr is None:
        raise HTTPException(status_code=503, detail="ASR 模型未加载，请先加载 TTS 模型: POST /model/load {\"model\": \"tts\"}")


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
- **语音克隆**：使用参考音频克隆音色和情感
- **批量配音**：批量生成多段配音
- **STT**：语音转文本

### 使用说明
1. 语音克隆需要提供 `text`（目标文本）、`ref_audio`（参考音频路径）、`ref_text`（参考音频文本）
2. 批量配音支持一次提交多段配音，自动合并
3. STT 使用 Whisper 模型进行语音识别
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


class STTRequest(BaseModel):
    """语音转文本请求"""
    ref_audio: str = Field(..., min_length=1)


class ModelLoadRequest(BaseModel):
    model: str = Field(..., pattern="^(tts|voxcpm2)$")


class VoxCloneRequest(BaseModel):
    """VoxCPM2 声音克隆请求"""
    text: str = Field(..., min_length=1, max_length=5000)
    ref_audio: str = Field(..., min_length=1)
    ref_text: Optional[str] = Field(None, min_length=1, max_length=5000)
    instruct: Optional[str] = Field(None, min_length=1, max_length=500)
    inference_timesteps: int = Field(5, ge=1, le=10)
    cfg_value: float = Field(3.0, ge=0.5, le=5.0)
    save_file: bool = True            # False=直接返回 WAV 流


class VoxDesignRequest(BaseModel):
    """VoxCPM2 声音设计请求"""
    text: str = Field(..., min_length=1, max_length=5000)
    instruct: str = Field(..., min_length=1, max_length=500)
    inference_timesteps: int = Field(7, ge=1, le=10)
    cfg_value: float = Field(3.0, ge=0.5, le=5.0)
    save_file: bool = True            # False=直接返回 WAV 流


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "qwen3_loaded": _qwen_tts is not None,
        "whisper_loaded": _qwen_asr is not None,
        "voxcpm2_loaded": _vox is not None,
    }


@app.get("/model-info")
async def model_info():
    return {
        "qwen3": {
            "path": TTS_MODEL_PATH,
            "loaded": _qwen_tts is not None,
            "has_asr_injected": _qwen_tts is not None and _qwen_tts.asr_model is not None,
        },
        "whisper": {
            "path": ASR_MODEL_PATH,
            "loaded": _qwen_asr is not None,
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
        "whisper": _qwen_asr is not None,
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
async def model_unload(request: ModelLoadRequest):
    if request.model == "tts":
        unload_qwen3()
    elif request.model == "voxcpm2":
        unload_vox()
    return {"success": True, "model": request.model, "action": "unloaded"}


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
                detail=f"音频生成失败（ref_audio={request.ref_audio}，请检查参考音频是否可访问、STT 识别是否成功）"
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
    
    - **items**: 配音列表，每项包含 text、ref_audio、ref_text
    - **merge**: 是否合并所有音频为一个文件
    - **output_filename**: 合并后的文件名
    
    Returns:
        - **success**: 是否成功
        - **count**: 成功生成的段数
        - **files**: 各段音频的文件路径列表
        - **merged_url**: 合并后的音频路径（如果 merge=True）
    """
    require_qwen3()
    
    if not request.items:
        raise HTTPException(status_code=400, detail="配音列表不能为空")
    
    results = []
    audio_list = []
    
    try:
        for i, item in enumerate(request.items):
            try:
                ref_path = _validate_audio_path(item.ref_audio)
            except HTTPException:
                logger.warning("第 %d 项参考音频路径不合法，跳过: %s", i + 1, item.ref_audio)
                continue
            if not ref_path.exists():
                logger.warning("第 %d 项参考音频不存在，跳过: %s", i + 1, item.ref_audio)
                continue
            
            filename = f"batch_{uuid.uuid4().hex[:8]}_{i+1:02d}.wav"
            output_path = OUTPUT_DIR / filename
            
            audio = _qwen_tts.generate(
                text=item.text,
                ref_audio=str(ref_path),
                ref_text=item.ref_text,
                stream=item.stream,
                output_path=str(output_path),
            )
            
            if audio is not None:
                results.append({
                    "index": i,
                    "text": item.text,
                    "audio_url": f"/output/{filename}",
                    "filename": filename,
                })
                audio_list.append(audio)
        
        response = {
            "success": True,
            "total": len(request.items),
            "generated": len(results),
            "files": results,
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
    
    Returns:
        - **success**: 是否成功
        - **audio_url**: 生成的音频文件路径
    """
    require_qwen3()
    
    if not request.items:
        raise HTTPException(status_code=400, detail="对话列表不能为空")
    
    try:
        dialogue_items = [
            {
                "text": item.text,
                "ref_audio": item.ref_audio,
                "ref_text": item.ref_text,
                "stream": item.stream,
            }
            for item in request.items
        ]
        
        audio_list = _qwen_tts.batch_generate(
            items=dialogue_items,
            output_dir=str(OUTPUT_DIR),
            save_individual=False,  # 只保存合并文件
        )
        
        # 过滤 None
        valid_audios = [a for a in audio_list if a is not None]
        
        if not valid_audios:
            raise HTTPException(status_code=500, detail="所有配音生成失败")
        
        if request.return_raw:
            merged = _merge_audio_arrays(valid_audios, 24000)
            if merged is not None:
                return Response(content=audio_to_wav_bytes(merged, 24000), media_type="audio/wav")

        # 合并保存
        output_path = OUTPUT_DIR / f"{request.output_filename}.wav"
        merge_audio_list(valid_audios, output_path, verbose=False)
        
        return {
            "success": True,
            "total_items": len(request.items),
            "generated": len(valid_audios),
            "audio_url": f"/output/{request.output_filename}.wav",
            "filename": f"{request.output_filename}.wav",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")


# ============================================================
# STT 接口（语音转文本）
# ============================================================

@app.post("/stt")
async def speech_to_text(request: STTRequest):
    """
    语音转文本接口
    
    使用 Whisper 模型将音频转换为文字
    
    - **ref_audio**: 音频文件路径
    
    Returns:
        - **text**: 识别出的文本
    """
    require_asr()
    
    audio_path = _validate_audio_path(request.ref_audio)
    if not audio_path.exists():
        raise HTTPException(status_code=400, detail=f"音频文件不存在: {request.ref_audio}")
    
    try:
        # mlx_audio.stt API: model.generate(audio_path) → result.text
        result = _qwen_asr.generate(str(audio_path))
        
        text = result.text if hasattr(result, 'text') else str(result)
        
        return {
            "success": True,
            "ref_audio": request.ref_audio,
            "text": text.strip(),
        }
        
    except Exception as e:
        logger.exception("Unhandled error in endpoint")
        raise HTTPException(status_code=500, detail=f"STT 转换失败: {str(e)}")


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
    logger.info("TTS 模型: %s", TTS_MODEL_PATH)
    logger.info("ASR 模型: %s", ASR_MODEL_PATH)
    logger.info("VoxCPM2: %s", VOX_MODEL_PATH)
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("API 文档: http://%s:%d/docs", host, port)
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level=log_level)
