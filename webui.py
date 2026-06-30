"""
TTS-Serve-MLX Web UI
基于 Gradio 的图形界面，调用本地 API 服务
启动 Web UI 时会自动拉起 API 服务，关闭时自动终止
"""
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import httpx

API_BASE = os.environ.get("TTS_SERVE_API_URL", "http://localhost:8000")
OUTPUT_DIR = Path("./api_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_api_process = None


def start_api_server():
    global _api_process
    if os.environ.get("TTS_SERVE_AUTO_START_API", "1") != "1":
        return False

    try:
        r = httpx.get(f"{API_BASE}/health", timeout=2)
        r.raise_for_status()
        return True
    except Exception:
        pass

    project_dir = Path(__file__).resolve().parent
    server_script = project_dir / "server_main.py"

    try:
        _api_process = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(project_dir),
        )
        return True
    except Exception as e:
        print(f"[webui] 启动 API 服务失败: {e}")
        return False


def wait_for_api(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_api_server():
    global _api_process
    if _api_process is not None:
        try:
            _api_process.send_signal(signal.SIGTERM)
            _api_process.wait(timeout=10)
        except Exception:
            _api_process.kill()
        _api_process = None


# ============================================================
# 模型管理
# ============================================================

def load_model(model_name):
    try:
        r = httpx.post(f"{API_BASE}/model/load", json={"model": model_name}, timeout=600)
        r.raise_for_status()
        return f"{model_name} 加载成功"
    except Exception as e:
        return f"加载失败: {e}"


def unload_model(model_name):
    try:
        r = httpx.post(f"{API_BASE}/model/unload", json={"model": model_name}, timeout=30)
        r.raise_for_status()
        return f"{model_name} 已卸载"
    except Exception as e:
        return f"卸载失败: {e}"


def model_status_text():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        d = r.json()
        qw = "加载" if d.get("qwen3_loaded") else "未加载"
        wh = "加载" if d.get("whisper_loaded") else "未加载"
        vx = "加载" if d.get("voxcpm2_loaded") else "未加载"
        return f"Qwen3: {qw} | Whisper: {wh} | VoxCPM2: {vx}"
    except Exception as e:
        return f"API 连接失败: {e}"


# ============================================================
# Tab 1: 语音克隆
# ============================================================

def clone_voice(text, ref_audio, ref_text, stream):
    if not text.strip():
        return None, "请输入要转换的文本"
    if ref_audio is None:
        return None, "请上传参考音频"

    ref_path = Path(ref_audio)
    payload = {
        "text": text.strip(),
        "ref_audio": str(ref_path),
        "ref_text": ref_text.strip() if ref_text else None,
        "stream": stream,
    }

    try:
        r = httpx.post(f"{API_BASE}/clone", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        local_path = str(OUTPUT_DIR / data["filename"])
        return local_path, f"生成成功: {data['filename']}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return None, "模型未加载，请先点击上方「加载 Qwen3 TTS」按钮"
        return None, f"请求失败 ({e.response.status_code}): {e.response.text[:100]}"
    except Exception as e:
        return None, f"生成失败: {e}"


# ============================================================
# Tab 2: 批量 TTS
# ============================================================



# ============================================================
# Tab 3: 语音转文本
# ============================================================

def transcribe_speech(ref_audio):
    if ref_audio is None:
        return "请上传音频文件"

    ref_path = Path(ref_audio)
    try:
        r = httpx.post(
            f"{API_BASE}/stt",
            json={"ref_audio": str(ref_path)},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("text", "")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return "模型未加载，请先点击上方「加载 Qwen3 TTS」按钮（ASR 随 TTS 一起加载）"
        return f"请求失败 ({e.response.status_code})"
    except Exception as e:
        return f"转录失败: {e}"


# ============================================================
# VoxCPM2 生成
# ============================================================

def vox_clone(text, ref_audio, ref_text, instruct, timesteps, cfg):
    if not text.strip():
        return None, "请输入要转换的文本"
    if ref_audio is None:
        return None, "请上传参考音频"

    payload = {
        "text": text.strip(),
        "ref_audio": str(Path(ref_audio)),
        "inference_timesteps": int(timesteps),
        "cfg_value": float(cfg),
    }
    if ref_text and ref_text.strip():
        payload["ref_text"] = ref_text.strip()
    if instruct and instruct.strip():
        payload["instruct"] = instruct.strip()

    try:
        r = httpx.post(f"{API_BASE}/vox/clone", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        local_path = str(OUTPUT_DIR / data["filename"])
        info = (
            f"生成成功 (克隆+情感)\n"
            f"处理: {data.get('processing_time', 0):.1f}s  "
            f"RTF: {data.get('real_time_factor', 0):.2f}x  "
            f"音频: {data.get('audio_duration', '')}"
        )
        return local_path, info
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return None, "VoxCPM2 模型未加载，请先点击上方「加载 VoxCPM2」按钮"
        return None, f"请求失败 ({e.response.status_code})"
    except Exception as e:
        return None, f"生成失败: {e}"


def vox_design(text, instruct, timesteps, cfg):
    if not text.strip():
        return None, "请输入要转换的文本"
    if not instruct.strip():
        return None, "请输入声音描述"

    payload = {
        "text": text.strip(),
        "instruct": instruct.strip(),
        "inference_timesteps": int(timesteps),
        "cfg_value": float(cfg),
    }

    try:
        r = httpx.post(f"{API_BASE}/vox/design", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        local_path = str(OUTPUT_DIR / data["filename"])
        info = (
            f"生成成功 (声音设计)\n"
            f"处理: {data.get('processing_time', 0):.1f}s  "
            f"RTF: {data.get('real_time_factor', 0):.2f}x  "
            f"音频: {data.get('audio_duration', '')}"
        )
        return local_path, info
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            return None, "VoxCPM2 模型未加载，请先点击上方「加载 VoxCPM2」按钮"
        return None, f"请求失败 ({e.response.status_code})"
    except Exception as e:
        return None, f"生成失败: {e}"


# ============================================================
# Gradio UI
# ============================================================

css = """
footer {display: none !important;}
#status_bar {text-align: center; padding: 8px; font-size: 13px;}
.model_row {margin-bottom: 10px;}
"""

with gr.Blocks(
    title="TTS-Serve-MLX",
    theme=gr.themes.Soft(),
    css=css,
) as demo:
    gr.Markdown("# TTS-Serve-MLX 语音合成")

    # ---- 模型管理工具栏 ----
    status = gr.Markdown(elem_id="status_bar")

    with gr.Row(elem_classes="model_row"):
        with gr.Column(scale=1):
            load_qwen_btn = gr.Button("加载 Qwen3 TTS", variant="primary", size="sm")
        with gr.Column(scale=1):
            unload_qwen_btn = gr.Button("卸载 Qwen3 TTS", variant="secondary", size="sm")
        with gr.Column(scale=1):
            load_vox_btn = gr.Button("加载 VoxCPM2", variant="primary", size="sm")
        with gr.Column(scale=1):
            unload_vox_btn = gr.Button("卸载 VoxCPM2", variant="secondary", size="sm")

    load_qwen_btn.click(fn=lambda: load_model("tts"), outputs=[status])
    unload_qwen_btn.click(fn=lambda: unload_model("tts"), outputs=[status])
    load_vox_btn.click(fn=lambda: load_model("voxcpm2"), outputs=[status])
    unload_vox_btn.click(fn=lambda: unload_model("voxcpm2"), outputs=[status])

    with gr.Tabs():
        # ---- Tab 1: 语音克隆 ----
        with gr.TabItem("语音克隆 (Qwen3 TTS)"):
            with gr.Row():
                with gr.Column(scale=2):
                    clone_text = gr.Textbox(
                        label="目标文本",
                        placeholder="输入要生成的文本...",
                        lines=4,
                    )
                    with gr.Row():
                        clone_ref_text = gr.Textbox(
                            label="参考音频文本 (可选，留空自动识别)",
                            placeholder="参考音频说了什么？",
                        )
                    clone_stream = gr.Checkbox(label="流式输出", value=False)
                    clone_btn = gr.Button("生成语音", variant="primary", scale=0)

                with gr.Column(scale=1):
                    clone_audio_input = gr.Audio(
                        label="参考音频",
                        type="filepath",
                        sources=["upload"],
                        editable=False,
                    )
                    clone_audio_output = gr.Audio(
                        label="生成结果",
                        type="filepath",
                        interactive=False,
                    )
                    clone_info = gr.Textbox(label="状态", interactive=False)

            clone_btn.click(
                fn=clone_voice,
                inputs=[clone_text, clone_audio_input, clone_ref_text, clone_stream],
                outputs=[clone_audio_output, clone_info],
            )

        # ---- Tab: 语音转文本 ----
        with gr.TabItem("语音转文本"):
            with gr.Row():
                with gr.Column(scale=1):
                    stt_audio = gr.Audio(
                        label="上传音频",
                        type="filepath",
                        sources=["upload"],
                    )
                    stt_btn = gr.Button("开始转录", variant="primary")
                with gr.Column(scale=2):
                    stt_result = gr.Textbox(
                        label="转录结果",
                        lines=10,
                        interactive=False,
                    )

            stt_btn.click(
                fn=transcribe_speech,
                inputs=[stt_audio],
                outputs=[stt_result],
            )

        # ---- Tab: VoxCPM2 克隆+情感 ----
        with gr.TabItem("VoxCPM2 克隆+情感"):
            with gr.Row():
                with gr.Column(scale=2):
                    vx_emo_text = gr.Textbox(
                        label="目标文本",
                        placeholder="输入要生成的文本...",
                        lines=4,
                    )
                    vx_emo_ref_text = gr.Textbox(
                        label="参考音频文本 (可选)",
                        placeholder="参考音频说了什么？",
                    )
                    vx_emo_instruct = gr.Textbox(
                        label="情感描述",
                        placeholder="例如: 非常开心、兴奋激动、语调上扬",
                        lines=2,
                    )
                    with gr.Row():
                        vx_emo_ts = gr.Slider(
                            label="推理步数", minimum=1, maximum=10, value=5, step=1,
                        )
                        vx_emo_cfg = gr.Slider(
                            label="CFG 强度", minimum=0.5, maximum=5.0, value=3.0, step=0.5,
                        )
                    vx_emo_btn = gr.Button("生成语音", variant="primary")

                with gr.Column(scale=1):
                    vx_emo_audio_in = gr.Audio(
                        label="参考音频",
                        type="filepath",
                        sources=["upload"],
                        editable=False,
                    )
                    vx_emo_audio_out = gr.Audio(
                        label="生成结果",
                        type="filepath",
                        interactive=False,
                    )
                    vx_emo_info = gr.Textbox(label="状态", interactive=False)

            vx_emo_btn.click(
                fn=lambda t, a, rt, i, ts, cfg: vox_clone(t, a, rt, i, ts, cfg),
                inputs=[vx_emo_text, vx_emo_audio_in, vx_emo_ref_text, vx_emo_instruct, vx_emo_ts, vx_emo_cfg],
                outputs=[vx_emo_audio_out, vx_emo_info],
            )

        # ---- Tab: VoxCPM2 声音设计 ----
        with gr.TabItem("VoxCPM2 声音设计"):
            with gr.Row():
                with gr.Column(scale=2):
                    vx_vd_text = gr.Textbox(
                        label="目标文本",
                        placeholder="输入要生成的文本...",
                        lines=4,
                    )
                    vx_vd_instruct = gr.Textbox(
                        label="声音描述",
                        placeholder="例如: 年轻温柔的女声，语调柔和亲切",
                        lines=3,
                    )
                    with gr.Row():
                        vx_vd_ts = gr.Slider(
                            label="推理步数", minimum=1, maximum=10, value=7, step=1,
                        )
                        vx_vd_cfg = gr.Slider(
                            label="CFG 强度", minimum=0.5, maximum=5.0, value=3.0, step=0.5,
                        )
                    vx_vd_btn = gr.Button("生成语音", variant="primary")

                with gr.Column(scale=1):
                    vx_vd_audio_out = gr.Audio(
                        label="生成结果",
                        type="filepath",
                        interactive=False,
                    )
                    vx_vd_info = gr.Textbox(label="状态", interactive=False)

            vx_vd_btn.click(
                fn=lambda t, i, ts, cfg: vox_design(t, i, ts, cfg),
                inputs=[vx_vd_text, vx_vd_instruct, vx_vd_ts, vx_vd_cfg],
                outputs=[vx_vd_audio_out, vx_vd_info],
            )

    demo.load(fn=model_status_text, outputs=[status])


if __name__ == "__main__":
    atexit.register(stop_api_server)

    started = start_api_server()
    if started and _api_process is not None:
        print(f"[webui] API 服务已启动 (PID {_api_process.pid})")
    elif started:
        print(f"[webui] 检测到已有 API 服务: {API_BASE}")
    else:
        print(f"[webui] 警告: 未能启动 API 服务")

    ready = wait_for_api()
    if not ready:
        print(f"[webui] 错误: API 服务在 180s 内未就绪")
        sys.exit(1)

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
