"""
Whisper STT 模块
用于将参考音频转换为文字，支持段落级和词级时间戳

使用方法：
    from stt import WhisperSTT, get_ref_audio_text
    
    # 方式1：使用便捷函数
    text = get_ref_audio_text("audio.wav")
    
    # 方式2：使用类（支持更多选项）
    stt = WhisperSTT(model_path="./models/whisper_asr_MLX")
    result = stt.transcribe("audio.wav")
    print(result.text)
    
    # 获取带时间戳的结果
    result = stt.transcribe("audio.wav", return_timestamps=True)
    for segment in result.segments:
        print(f"[{segment['start']:.2f}s] {segment['text']}")
"""

from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据类
# ============================================================

@dataclass
class TranscriptionResult:
    """转录结果数据类"""
    text: str                           # 完整转录文本
    language: Optional[str] = None      # 检测到的语言
    segments: Optional[List[Dict]] = None  # 段落列表（带时间戳时）
    
    @property
    def has_timestamps(self) -> bool:
        """是否有时间戳"""
        return self.segments is not None


# ============================================================
# WhisperSTT 类
# ============================================================

class WhisperSTT:
    """
    Whisper 语音转文字封装类
    
    Attributes:
        model_path: 模型路径或 HuggingFace 仓库名
        model: 加载的模型实例
    
    Example:
        stt = WhisperSTT("./models/whisper_asr_MLX")
        
        # 基本转录
        result = stt.transcribe("audio.wav")
        print(result.text)
        
        # 带时间戳
        result = stt.transcribe("audio.wav", return_timestamps=True)
        for seg in result.segments:
            print(f"[{seg['start']:.1f}s] {seg['text']}")
        
        # 词级时间戳
        result = stt.transcribe("audio.wav", word_timestamps=True)
    """
    
    def __init__(self, model_path: str = "./models/whisper_asr_MLX"):
        from mlx_audio.stt import load as load_stt
        
        self.model_path = model_path
        self._model = None
        self._load_model()
    
    def _load_model(self):
        from mlx_audio.stt import load as load_stt
        
        logger.info("正在加载 Whisper 模型: %s", self.model_path)
        self._model = load_stt(self.model_path)
        logger.info("Whisper 模型加载完成")
    
    @property
    def model(self):
        """获取模型实例"""
        return self._model
    
    def transcribe(
        self,
        audio_path: str,
        return_timestamps: bool = True,
        word_timestamps: bool = False,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            return_timestamps: 是否返回段落级时间戳
            word_timestamps: 是否返回词级时间戳
            language: 强制指定语言（None=自动检测）
        
        Returns:
            TranscriptionResult 对象
        
        Raises:
            FileNotFoundError: 音频文件不存在
            Exception: 转录失败
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            # 调用模型生成
            result = self._model.generate(
                str(audio_path),
                return_timestamps=return_timestamps,
                word_timestamps=word_timestamps,
                language=language,
            )
            
            # 提取结果
            text = result.text if hasattr(result, 'text') else str(result)
            language = getattr(result, 'language', None)
            segments = getattr(result, 'segments', None)
            
            return TranscriptionResult(
                text=text.strip() if text else "",
                language=language,
                segments=segments,
            )
            
        except Exception as e:
            raise Exception(f"转录失败: {e}")
    
    def transcribe_simple(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        简单转录，只返回文本
        
        Args:
            audio_path: 音频文件路径
            language: 强制指定语言
        
        Returns:
            转录文本
        """
        result = self.transcribe(
            audio_path,
            return_timestamps=False,
            language=language,
        )
        return result.text
    
    def transcribe_with_segments(self, audio_path: str) -> TranscriptionResult:
        """
        转录并返回段落级时间戳
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            TranscriptionResult（包含 segments）
        """
        return self.transcribe(audio_path, return_timestamps=True)
    
    def transcribe_with_words(self, audio_path: str) -> TranscriptionResult:
        """
        转录并返回词级时间戳
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            TranscriptionResult（包含 words）
        """
        return self.transcribe(
            audio_path,
            return_timestamps=True,
            word_timestamps=True,
        )
    
    def format_verbose(self, result: TranscriptionResult) -> str:
        """
        格式化输出（带时间戳）
        
        Args:
            result: 转录结果
        
        Returns:
            格式化后的字符串
        """
        if not result.has_timestamps:
            return result.text
        
        lines = []
        for segment in result.segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
            lines.append(f"[{start:.2f}s -> {end:.2f}s] {text}")
        
        return '\n'.join(lines)
    
    def unload(self):
        self._model = None


# ============================================================
# 便捷函数
# ============================================================

def get_ref_audio_text(
    audio_path: str,
    model_path: Optional[str] = None,
    language: Optional[str] = None,
    asr_model=None,
) -> str:
    """
    获取参考音频的转录文本（便捷函数）

    Args:
        audio_path: 参考音频文件路径
        model_path: Whisper 模型路径（默认从 TTS_SERVE_MODELS_DIR 读取）
        language: 强制指定语言（None=自动检测）
        asr_model: 已加载的 ASR 模型实例（优先使用，避免重复加载）

    Returns:
        转录文本
    """
    if asr_model is not None:
        return asr_model.transcribe_simple(audio_path, language=language)
    stt = get_stt(model_path=model_path)
    return stt.transcribe_simple(audio_path, language=language)


def transcribe_audio(
    audio_path: str,
    model_path: Optional[str] = None,
    return_timestamps: bool = True,
    asr_model=None,
) -> TranscriptionResult:
    """
    转录音频（完整功能）

    Args:
        audio_path: 音频文件路径
        model_path: 模型路径（默认从 TTS_SERVE_MODELS_DIR 读取）
        return_timestamps: 是否返回时间戳
        asr_model: 已加载的 ASR 模型实例（优先使用）

    Returns:
        TranscriptionResult 对象
    """
    if asr_model is not None:
        return asr_model.transcribe(audio_path, return_timestamps=return_timestamps)
    stt = get_stt(model_path=model_path)
    return stt.transcribe(audio_path, return_timestamps=return_timestamps)


# ============================================================
# 模块初始化
# ============================================================

import os as _os

_DEFAULT_ASR_PATH = str(
    Path(_os.environ.get("TTS_SERVE_MODELS_DIR", "./models")) / "whisper_asr_MLX"
)

# 全局单例（延迟加载）
_stt_instance: Optional[WhisperSTT] = None


def get_stt(model_path: Optional[str] = None) -> WhisperSTT:
    """
    获取 WhisperSTT 单例实例

    Args:
        model_path: 模型路径（默认从 TTS_SERVE_MODELS_DIR 环境变量读取）

    Returns:
        WhisperSTT 实例
    """
    global _stt_instance
    if model_path is None:
        model_path = _DEFAULT_ASR_PATH

    if _stt_instance is None:
        _stt_instance = WhisperSTT(model_path=model_path)

    return _stt_instance


# ============================================================
# 测试
# ============================================================

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("用法: python stt.py <音频文件路径>")
#         print("示例: python stt.py audio.wav")
#         sys.exit(1)
    
#     audio_path = sys.argv[1]
    
#     print("=" * 60)
#     print("Whisper STT 转录工具")
#     print("=" * 60)
    
#     try:
#         # 使用便捷函数
#         text = get_ref_audio_text(audio_path)
#         print(f"\n转录结果:\n{text}")
        
#         # 或者使用完整功能
#         print("\n" + "-" * 60)
#         print("带时间戳结果:")
#         print("-" * 60)
        
#         stt = get_stt()
#         result = stt.transcribe_with_segments(audio_path)
#         print(stt.format_verbose(result))
        
#     except FileNotFoundError as e:
#         print(f"\n✗ 错误: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\n✗ 转录失败: {e}")
#         sys.exit(1)
# text = get_ref_audio_text('./【默认】为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。.wav')
# print("转录结果:", text)