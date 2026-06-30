"""
VoxCPM2 测试脚本
语音克隆 + 情感模式，遍历 [steps, cfg] 参数组合
"""
import time
from pathlib import Path
import numpy as np
from mlx_audio.tts.utils import load

OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

REF_AUDIO = "./【开心_happy】等等，停云小姐，你怎么还不回去！这里离丹炉很近了！.wav"
TEXT = "为了他的命，也为了我们自己，得想办法把他抓回来。免得更麻烦！"
EMOTION = "难过"

t_start = time.time()

print("=" * 60)
print("VoxCPM2 测试脚本")
print("模式: 语音克隆 + 情感控制")
print(f"参考音频: {REF_AUDIO}")
print(f"情感指令: {EMOTION}")
print("=" * 60)

t1 = time.time()
model = load("./models/voxCPM2_4bit_MLX")
print(f"\n模型加载: {time.time() - t1:.1f}s\n")


def save_audio(audio_array, name, sample_rate=48000):
    import soundfile as sf
    path = OUTPUT_DIR / name
    sf.write(str(path), np.array(audio_array), sample_rate)
    print(f"  已保存: {path.name}")


combos = [(7, 3), (7, 2), (5, 3), (5, 2)]
for steps, cfg in combos:
    t2 = time.time()
    result = next(model.generate(
        text=TEXT,
        ref_audio=REF_AUDIO,
        instruct=EMOTION,
        inference_timesteps=steps,
        cfg_value=cfg,
    ))
    save_audio(result.audio, f"vox_{steps}_{cfg}.wav")
    pt = getattr(result, "processing_time_seconds", "?")
    rtf = getattr(result, "real_time_factor", "?")
    dur = getattr(result, "audio_duration", "?")
    print(f"  [steps={steps}, cfg={cfg}]  处理 {pt}s | RTF {rtf} | 音频 {dur} | wall {time.time() - t2:.1f}s")

model = None
import gc
gc.collect()

print(f"\n完成！共 {len(combos)} 个音频文件")
print(f"输出: {OUTPUT_DIR.resolve()}")
print(f"总耗时: {time.time() - t_start:.1f}s")
