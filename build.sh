#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TTS-Serve-MLX 打包脚本
# ============================================================
# 用法:
#   source .venv/bin/activate && ./build.sh
#
# 产物: dist/tts_serve_mlx/
#   ├── tts_serve_mlx       ← 可执行文件
#   ├── _internal/          ← Python 运行时
#   └── models/             ← 空结构，放置模型文件
#       ├── qwenTTS_0.6B_MLX/
#       ├── whisper_asr_MLX/
#       └── voxCPM2_4bit_MLX/
#
# 前置条件:
#   - Python 3.14 (arm64)，虚拟环境已安装依赖
#   - brew install libsndfile
#   - 模型文件已下载至 models/ 目录（仅用于验证）
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

DIST_DIR="$PROJECT_DIR/dist/tts_serve_mlx"
MODELS_OUT="$DIST_DIR/models"

echo "========================================"
echo " TTS-Serve-MLX 打包工具"
echo "========================================"
echo "项目目录: $PROJECT_DIR"
echo "Python:   $(python3 --version)"
echo ""

# ---- 检查依赖 ----
echo "[1/5] 检查依赖..."
python3 -c "import fastapi, uvicorn, soundfile, numpy, mlx, mlx_audio, tts_clone, stt" 2>/dev/null || {
    echo "[!] 部分依赖缺失，正在安装..."
    pip install -r requirements.txt
}
echo "[✓] 依赖检查完成"
echo ""

# ---- 检查 libsndfile ----
echo "[2/5] 检查 libsndfile..."
LIBSNDFILE="/opt/homebrew/lib/libsndfile.dylib"
if [ ! -f "$LIBSNDFILE" ]; then
    echo "[!] libsndfile 未找到，请安装: brew install libsndfile"
    exit 1
fi
echo "[✓] libsndfile: $LIBSNDFILE"
echo ""

# ---- 检查模型文件（仅验证存在，不打包） ----
echo "[3/5] 检查模型文件（验证用，不打包）..."
MODELS=(
    "models/qwenTTS_0.6B_MLX"
    "models/whisper_asr_MLX"
    "models/voxCPM2_4bit_MLX"
)
for d in "${MODELS[@]}"; do
    if [ ! -d "$PROJECT_DIR/$d" ]; then
        echo "[!] 模型目录不存在: $PROJECT_DIR/$d"
        exit 1
    fi
    count=$(ls "$PROJECT_DIR/$d"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    echo "    $d ($count .safetensors 文件)"
done
echo "[✓] 模型文件检查完成"
echo ""

# ---- 清理旧构建 ----
echo "[4/5] 清理旧构建..."
rm -rf "$PROJECT_DIR/build" "$DIST_DIR" 2>/dev/null || true
echo "[✓] 清理完成"
echo ""

# ---- 执行 PyInstaller 打包 ----
echo "[5/5] 开始 PyInstaller 打包..."
echo "    模式: onedir"
echo "    目标: arm64"
echo "    注意: 模型文件不打包，用户自行复制到 dist/tts_serve_mlx/models/"
echo ""

python3 -m PyInstaller \
  --onedir \
  --name tts_serve_mlx \
  --add-binary "$LIBSNDFILE:." \
  --hidden-import "api" \
  --hidden-import "tts_clone" \
  --hidden-import "stt" \
  --hidden-import "mlx_audio.tts.utils" \
  --hidden-import "mlx_audio.tts.models.qwen3_tts" \
  --hidden-import "mlx_audio.stt.utils" \
  --hidden-import "mlx_audio.stt.models.whisper" \
  --hidden-import "mlx_lm" \
  --collect-all "mlx_audio" \
  --collect-all "mlx" \
  --collect-all "mlx_lm" \
  --collect-all "transformers" \
  --target-arch arm64 \
  server_main.py

echo ""

# ---- 创建空 models/ 目录结构 + 说明文件 ----
echo "创建模型目录结构 (空壳 + 说明)..."
for subdir in qwenTTS_0.6B_MLX whisper_asr_MLX voxCPM2_4bit_MLX; do
    mkdir -p "$MODELS_OUT/$subdir"
done

cat > "$MODELS_OUT/qwenTTS_0.6B_MLX/README.txt" << 'README'
模型: Qwen3-TTS 0.6B (4-bit)
来源: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
用途: 语音克隆、批量配音、对话生成（Speaker 模式，无需 ASR）

文件清单:
  - model.safetensors (主模型 ~977MB)
  - speech_tokenizer/model.safetensors (语音编解码器 ~651MB)
  - config.json, tokenizer_config.json, vocab.json 等

下载:
  HF:          huggingface-cli download mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit --local-dir ./models/qwenTTS_0.6B_MLX
  魔搭(无需代理): git clone https://www.modelscope.cn/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit.git ./models/qwenTTS_0.6B_MLX
README

cat > "$MODELS_OUT/whisper_asr_MLX/README.txt" << 'README'
模型: Whisper Large v3 Turbo ASR (fp16)
来源: mlx-community/whisper-large-v3-turbo-asr-fp16
用途: 语音转文本（独立加载，不与 TTS 绑定）

文件清单:
  - model.safetensors (主模型 ~1.5GB)
  - config.json, tokenizer.json, vocab.json 等

下载:
  HF:          huggingface-cli download mlx-community/whisper-large-v3-turbo-asr-fp16 --local-dir ./models/whisper_asr_MLX
  魔搭(无需代理): git clone https://www.modelscope.cn/NexaAIDev/whisper-large-v3-turbo-MLX.git ./models/whisper_asr_MLX
README

cat > "$MODELS_OUT/voxCPM2_4bit_MLX/README.txt" << 'README'
模型: VoxCPM2 2B (4-bit)
来源: mlx-community/VoxCPM2-4bit
用途: 情感克隆（steps=6, cfg=4.0）、声音设计
特点: 48kHz 输出，原生 instruct 支持

文件清单:
  - model.safetensors (主模型 ~2.1GB)
  - config.json, tokenizer.json 等

下载:
  HF:          huggingface-cli download mlx-community/VoxCPM2-4bit --local-dir ./models/voxCPM2_4bit_MLX
  魔搭(无需代理): git clone https://www.modelscope.cn/aufklarer/VoxCPM2-MLX-int4.git ./models/voxCPM2_4bit_MLX
README

echo ""
echo "models/ 目录已创建:"
find "$MODELS_OUT" -name "README.txt" | while read f; do
    dir=$(dirname "$f" | sed "s|$DIST_DIR/||")
    echo "    $dir/README.txt"
done

echo ""
echo "========================================"
echo "[✓] 打包完成！"
echo "========================================"
echo "产物路径:"
echo "  $DIST_DIR/"
echo ""
echo "使用方式:"
echo "  1. 将模型文件复制到:"
echo "     $DIST_DIR/models/qwenTTS_0.6B_MLX/"
echo "     $DIST_DIR/models/whisper_asr_MLX/"
echo "     $DIST_DIR/models/voxCPM2_4bit_MLX/"
echo "  2. 启动服务:"
echo "     $DIST_DIR/tts_serve_mlx"
echo ""
echo "自定义模型路径:"
echo "  TTS_SERVE_MODELS_DIR=/path/to/models $DIST_DIR/tts_serve_mlx"
echo ""
echo "Electron 集成:"
echo "  将 $DIST_DIR/ 复制到 Electron 项目"
echo "  resources/python-server/ 目录下，参考 ELECTRON_INTEGRATION.md"
echo "========================================"
