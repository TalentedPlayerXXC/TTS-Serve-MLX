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

# ---- 创建空 models/ 目录结构 ----
echo "创建模型目录结构 (空壳)..."
for subdir in qwenTTS_0.6B_MLX whisper_asr_MLX voxCPM2_4bit_MLX; do
    mkdir -p "$MODELS_OUT/$subdir"
    echo "将此目录下的模型文件 (.safetensors, .json 等) 复制到此处" > "$MODELS_OUT/$subdir/README.txt"
done
echo "[✓] models/ 目录已创建:"
find "$MODELS_OUT" -type d | sed "s|$DIST_DIR/|    |"

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
