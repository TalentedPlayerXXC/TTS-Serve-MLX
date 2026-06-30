"""
TTS-Serve-MLX FastAPI Server 启动入口
同时适配 PyInstaller 打包模式和直接 python 运行模式
"""
import logging
import os
import sys
import uvicorn


def main():
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    port = int(os.environ.get("TTS_SERVE_PORT", "8000"))
    host = os.environ.get("TTS_SERVE_HOST", "127.0.0.1")

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
