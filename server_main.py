"""
TTS-Serve-MLX FastAPI Server 启动入口
同时适配 PyInstaller 打包模式和直接 python 运行模式

端口发现机制（方便 Electron 集成）：
  - 设了 TTS_SERVE_PORT → 用指定值
  - 没设 → 找 8000-8050 范围空闲端口
  - 找到后 stdout 打印 TTS_SERVER_PORT=xxxxx，Electron 抓取解析
"""
import logging
import os
import socket
import sys
import uvicorn


def _find_free_port() -> int:
    """找一个空闲端口，优先 8000-8050 范围"""
    for port in range(8000, 8051):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    # 范围内全满，让 OS 分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main():
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    port = int(os.environ["TTS_SERVE_PORT"]) if "TTS_SERVE_PORT" in os.environ else _find_free_port()
    host = os.environ.get("TTS_SERVE_HOST", "127.0.0.1")
    models_dir = os.environ.get("TTS_SERVE_MODELS_DIR", "./models")

    logger = logging.getLogger(__name__)
    logger.info("启动配置: host=%s port=%s models_dir=%s", host, port, models_dir)

    # 打印端口号给父进程（Electron 主进程抓 stdout 解析）
    print(f"TTS_SERVER_PORT={port}", flush=True)

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        log_level=os.environ.get("TTS_SERVE_LOG_LEVEL", "warning"),
        reload=False,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
