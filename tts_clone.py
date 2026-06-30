"""
TTS-Serve 语音克隆模块
基于 mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 模型

功能：
- 使用参考音频进行语音克隆，保留原音频的音色和情感风格
- 支持多角色对话批量生成
- 内置音频后处理（归一化、淡入淡出、交叉淡入淡出拼接）
- 作为模块被 api.py 或其他文件导入调用

调用方式：
    from tts_clone import TTSClone

    tts = TTSClone(model_path="./models/qwenTTS_0.6B_MLX")

    # 单人配音（ref_text 不传则自动 STT 识别）
    audio = tts.generate("要转换的文本", "参考音频.wav")

    # 批量配音
    results = tts.batch_generate([
        {"text": "你好", "ref_audio": "ref1.wav"},
        {"text": "再见", "ref_audio": "ref2.wav"},
    ])

    # 多人配音（批量生成 + 合并为一个音频）
    results = tts.batch_generate([
        {"text": "我是张三", "ref_audio": "zhangsan.wav"},
        {"text": "我是李四", "ref_audio": "lisi.wav"},
    ])
    tts.merge_and_save(results, "dialogue.wav")
"""

from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import io
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 音频处理工具函数
# ============================================================

def save_audio(audio_array, output_path: Path, sample_rate: int = 24000, verbose: bool = True):
    """
    保存音频为 WAV 文件，带归一化和削波处理
    
    Args:
        audio_array: 音频数组
        output_path: 输出文件路径
        sample_rate: 采样率，默认 24000
        verbose: 是否打印保存信息
    """
    import soundfile as sf
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为 numpy 数组
    audio_array = np.asarray(audio_array)
    
    # 音频归一化和削波处理，防止破音
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        if max_val > 1.5:
            audio_array = audio_array / max_val * 0.95
            if verbose:
                logger.info("音频幅度过大(%.2f)，已自动归一化", max_val)
        else:
            audio_array = np.clip(audio_array, -0.98, 0.98)
    
    sf.write(str(output_path), audio_array.astype(np.float32), sample_rate)
    if verbose:
        logger.info("已保存: %s", output_path)


def audio_to_wav_bytes(audio_array, sample_rate: int = 24000) -> bytes:
    """将音频 numpy 数组转为 WAV 字节"""
    import soundfile as sf
    audio_array = np.asarray(audio_array).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format='WAV')
    return buf.getvalue()


def apply_fade(audio: np.ndarray, fade_in_samples: int = 240, fade_out_samples: int = 480) -> np.ndarray:
    """
    对音频应用淡入淡出处理
    
    Args:
        audio: 输入音频数组
        fade_in_samples: 淡入采样点数（约 10ms）
        fade_out_samples: 淡出采样点数（约 20ms）
    
    Returns:
        处理后的音频数组
    """
    audio = np.asarray(audio)
    length = len(audio)
    
    fade_in = np.linspace(0, 1, min(fade_in_samples, length // 4))
    fade_out = np.linspace(1, 0, min(fade_out_samples, length // 4))
    
    if length > len(fade_in):
        audio[:len(fade_in)] = audio[:len(fade_in)] * fade_in
    
    if length > len(fade_out):
        audio[-len(fade_out):] = audio[-len(fade_out):] * fade_out
    
    return audio


def _merge_audio_arrays(audio_list: List[np.ndarray], sample_rate: int = 24000,
                        silence_duration: float = 0.5) -> Optional[np.ndarray]:
    """合并多个音频数组为单个 numpy 数组（内存操作，不写磁盘）"""
    if not audio_list:
        return None

    crossfade_samples = int(sample_rate * 0.1)
    silence_samples = int(sample_rate * silence_duration)

    result = np.asarray(audio_list[0])
    result = apply_fade(result)

    for audio in audio_list[1:]:
        audio = apply_fade(np.asarray(audio))
        overlap = min(len(result), len(audio), crossfade_samples)

        if overlap > 0:
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            result[-overlap:] *= fade_out
            audio[:overlap] *= fade_in
            result = np.concatenate([result, audio[overlap:]])
        else:
            result = np.concatenate([result, audio])

        silence = np.zeros(silence_samples)
        result = np.concatenate([result, silence])

    return result


def merge_audio_list(audio_list: List[np.ndarray], output_path: Path, sample_rate: int = 24000, 
                      silence_duration: float = 0.5, verbose: bool = True) -> bool:
    """
    合并多个音频片段并保存为 WAV 文件
    """
    merged = _merge_audio_arrays(audio_list, sample_rate, silence_duration)
    if merged is None:
        return False
    save_audio(merged, output_path, sample_rate, verbose)
    return True


# ============================================================
# TTSClone 类 - 核心封装
# ============================================================

@dataclass
class TTSItem:
    """配音项目数据类"""
    text: str           # 要转换的文本
    ref_audio: str      # 参考音频路径
    ref_text: str       # 参考音频对应的文本


class TTSClone:
    """
    多模型 TTS 语音克隆封装类
    
    Attributes:
        model_path: 模型路径或 HuggingFace 模型 ID
        sample_rate: 音频采样率，默认 24000
    
    Example:
        # 基本用法
        tts = TTSClone("./models/qwenTTS_0.6B_MLX")
        
        # 单条生成
        audio = tts.generate("你好世界", "ref.wav", "参考文本")
        
        # 批量生成
        results = tts.batch_generate([
            {"text": "你好", "ref_audio": "ref1.wav", "ref_text": "文本1"},
            {"text": "再见", "ref_audio": "ref2.wav", "ref_text": "文本2"},
        ])
        
        # 合并保存
        tts.merge_and_save(results, "output.wav")
    """
    
    def __init__(self, model_path: str = "./models/qwenTTS_0.6B_MLX", sample_rate: int = 24000,
                 asr_model=None):
        """
        初始化 TTSClone
        
        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            sample_rate: 音频采样率
            asr_model: 已加载的 ASR 模型实例（用于 ref_text 转录，避免重复加载 Whisper）
        """
        from mlx_audio.tts.utils import load_model
        
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model = None
        self.asr_model = asr_model
    
    @property
    def model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model
            logger.info("正在加载模型: %s", self.model_path)
            self._model = load_model(self.model_path)
            logger.info("模型加载完成")
        return self._model
    
    def generate(self, text: str, ref_audio: str, ref_text: Optional[str] = None,
                 stream: bool = False,
                 output_path: Optional[str] = None, apply_fade_process: bool = True) -> Optional[np.ndarray]:
        """
        使用语音克隆生成单条音频
        
        Args:
            text: 要转换的文本
            ref_audio: 参考音频文件路径
            ref_text: 参考音频对应的文本（可选，不传则自动识别）
            output_path: 可选，输出文件路径
            apply_fade_process: 是否应用淡入淡出处理
        
        Returns:
            音频数组（numpy）或 None
        
        Example:
            # 方式1：自动识别 ref_text（推荐）
            audio = tts.generate(
                text="你好，很高兴见到你！",
                ref_audio="./ref.wav",
                output_path="./output.wav"
            )
            
            # 方式2：手动指定 ref_text
            audio = tts.generate(
                text="你好",
                ref_audio="./ref.wav",
                ref_text="这是参考音频原文。",
            )
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            logger.warning("参考音频不存在: %s", ref_audio)
            return None
        
        if not ref_text:
            logger.info("自动识别参考音频文字: %s", str(ref_path))
            try:
                if self.asr_model is not None:
                    ref_text = self.asr_model.transcribe_simple(str(ref_path))
                else:
                    from stt import get_ref_audio_text
                    ref_text = get_ref_audio_text(str(ref_path))
                logger.info("识别结果: %s", ref_text[:60] + ('...' if len(ref_text) > 60 else ''))
            except Exception as e:
                logger.error("参考音频文字识别失败: %s", e)
                logger.error("音频路径: %s", ref_audio)
                return None
        
        try:
            if stream:
                chunks = []
                for result in self.model.generate(
                    text=text,
                    ref_audio=str(ref_path),
                    ref_text=ref_text,
                    stream=True,
                    streaming_interval=0.32,
                ):
                    chunks.append(np.array(result.audio))
                if not chunks:
                    return None
                audio = np.concatenate(chunks)
            else:
                results = list(self.model.generate(
                    text=text,
                    ref_audio=str(ref_path),
                    ref_text=ref_text,
                    stream=False,
                ))
                if not results:
                    return None
                audio = np.array(results[0].audio)
            
            # 淡入淡出处理
            if apply_fade_process:
                audio = apply_fade(audio, fade_in_samples=240, fade_out_samples=480)
            
            # 保存文件
            if output_path:
                save_audio(audio, Path(output_path))
            
            return audio
            
        except Exception as e:
            logger.error("生成失败: %s", e)
            return None
    
    def batch_generate(self, items: List[Dict[str, str]], 
                       output_dir: Optional[str] = None,
                       save_individual: bool = True) -> List[Optional[np.ndarray]]:
        """
        批量生成多个配音
        
        Args:
            items: 配音列表，每个元素为 dict:
                - text: 要转换的文本（必填）
                - ref_audio: 参考音频路径（必填）
                - ref_text: 参考音频对应文本（可选，不传则自动识别）
            output_dir: 输出目录（可选）
            save_individual: 是否保存每个单独的音频文件
        
        Returns:
            音频数组列表（与输入顺序对应）
        
        Example:
            # ref_text 可选，不传自动识别
            results = tts.batch_generate([
                {"text": "你好", "ref_audio": "ref1.wav"},  # 自动识别
                {"text": "再见", "ref_audio": "ref2.wav", "ref_text": "参考2"},  # 手动指定
            ])
        """
        results = []
        output_path_base = Path(output_dir) if output_dir else None
        
        if output_path_base:
            output_path_base.mkdir(parents=True, exist_ok=True)
        
        logger.info("开始批量生成 %d 段配音...", len(items))
        
        for i, item in enumerate(items):
            text = item["text"]
            ref_audio = item["ref_audio"]
            ref_text = item.get("ref_text")
            stream = item.get("stream", False)
            
            logger.info("[%d/%d] %s", i + 1, len(items), text[:30] + ('...' if len(text) > 30 else ''))
            
            output_path = None
            if save_individual and output_path_base:
                output_path = output_path_base / f"clone_{i+1:02d}.wav"
            
            audio = self.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                stream=stream,
                output_path=str(output_path) if output_path else None,
            )
            
            results.append(audio)
        
        return results
    
    def merge_and_save(self, audio_list: List[np.ndarray], output_path: str,
                       silence_duration: float = 0.5) -> bool:
        """
        合并多个音频并保存
        
        Args:
            audio_list: 音频数组列表
            output_path: 输出文件路径
            silence_duration: 静音间隔（秒）
        
        Returns:
            是否成功
        """
        return merge_audio_list(
            audio_list, 
            Path(output_path), 
            self.sample_rate, 
            silence_duration
        )
    
    def generate_dialogue(self, dialogue_items: List[Dict], output_path: Optional[str] = None) -> List[np.ndarray]:
        """
        生成对话场景的配音（快捷方法）
        
        Args:
            dialogue_items: 对话列表，每个元素为 dict:
                - text: 要说的话（必填）
                - ref_audio: 参考音频（必填）
                - ref_text: 参考文本（可选，不传则自动识别）
            output_path: 可选，合并后的输出路径
        
        Returns:
            音频数组列表
        
        Example:
            # ref_text 可选
            audio_list = tts.generate_dialogue([
                {"text": "你好", "ref_audio": "ref1.wav"},           # 自动识别
                {"text": "你也好", "ref_audio": "ref2.wav", "ref_text": "参考2"},  # 手动指定
            ], output_path="dialogue.wav")
        """
        results = self.batch_generate(dialogue_items)
        
        if output_path and results:
            self.merge_and_save(results, output_path)
        
        return results
    
    def unload(self):
        """卸载模型，释放内存"""
        self._model = None
        import gc
        gc.collect()



