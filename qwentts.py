import os
from mlx_audio.tts.utils import load_model
# from mlx_audio.tts.generate import generate_audio
import soundfile as sf
import numpy as np


tts_model = load_model("./models/qwenTTS_0.6B_MLX")
ref_audio = "./【默认】为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。.wav"
ref_text  = "为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。"
# print("ref_text:", ref_text)

# texts = [
#     "本段语音基于千问TTS-MLX模型生成，",
#     "一下为生成的文本示例：",
#     "为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。",
# ]
# ref_audios = [
#     "./【默认】为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。.wav",
#     "./【开心_happy】嘿嘿~温情的诉说时间结束！.wav",
#     "./【开心_happy】等等，停云小姐，你怎么还不回去！这里离丹炉很近了！.wav",
# ]

# ref_texts = [
#     '为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。',
#     '嘿嘿~温情的诉说时间结束！',
#     '等等，停云小姐，你怎么还不回去！这里离丹炉很近了！',
# ]
results = list(tts_model.generate(
    text='本段语音基于千问TTS-MLX模型生成，一下为生成的文本示例：为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。',
    ref_audio=ref_audio,
    ref_text=ref_text
))

audio = results[0].audio  # mx.array
# 从 mx.array 转为 numpy
audio_np = np.array(audio)

# 尝试获取采样率，若没有则手动指定（例如 24000）
sample_rate = getattr(results[0], "sample_rate", 24000)
# 保存为 WAV
sf.write("output.wav", audio_np, sample_rate)
