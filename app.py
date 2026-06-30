"""
批量配音测试
使用项目中的三个参考音频进行语音克隆
"""
import time
from pathlib import Path
from tts_clone import TTSClone, merge_audio_list

t_start = time.time()

OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

items = [
    {
        "text": "你好，我是赞妮，是一个喜欢冒险的女孩，你是谁呀？",
        "ref_audio": "./【默认】为了他的命，也为了我们自己，得想办法把他抓回来，免得更麻烦。.wav",
    },
    {
        "text": "你好，我是花火，很高兴见到你！那你呢？",
        "ref_audio": "./【开心_happy】嘿嘿~温情的诉说时间结束！.wav",
    },
    {
        "text": "我是三月，喜欢冒险和美食！我们一起去吃火锅吧！",
        "ref_audio": "./【开心_happy】等等，停云小姐，你怎么还不回去！这里离丹炉很近了！.wav",
    },
]

t1 = time.time()
tts = TTSClone()
print(f"初始化: {time.time() - t1:.1f}s")

t2 = time.time()
results = tts.batch_generate(items, output_dir=str(OUTPUT_DIR))
print(f"生成耗时: {time.time() - t2:.1f}s")

valid = [r for r in results if r is not None]
print(f"成功: {len(valid)}/{len(results)} 段")

if valid:
    t3 = time.time()
    merged_path = OUTPUT_DIR / "merged.wav"
    merge_audio_list(valid, merged_path)
    print(f"合并耗时: {time.time() - t3:.1f}s")
    print(f"合并音频: {merged_path}")

tts.unload()
print(f"\n总耗时: {time.time() - t_start:.1f}s")