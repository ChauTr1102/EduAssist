import gradio as gr
import requests
import tempfile
import os
from typing import Optional

# Cấu hình API
API_URL = "http://localhost:8000"  # Thay đổi nếu API chạy ở URL khác


def call_api(endpoint: str, method: str = "get", params: Optional[dict] = None, files: Optional[dict] = None):
    """Hàm gọi API đến backend"""
    try:
        if method.lower() == "get":
            response = requests.get(f"{API_URL}/{endpoint}", params=params)
        elif method.lower() == "post":
            response = requests.post(f"{API_URL}/{endpoint}", params=params, files=files)
        else:
            return {"error": "Invalid method"}

        return response.json() if response.status_code == 200 else {"error": f"API error: {response.text}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def get_model_info():
    """Lấy thông tin model hiện tại"""
    return call_api("model-info")


def transcribe_audio(audio_path: str, model_name: str = "large-v3"):
    """Gửi audio đến API để chuyển đổi thành văn bản"""
    if not audio_path:
        return {"error": "Vui lòng chọn file audio"}

    try:
        with open(audio_path, "rb") as audio_file:
            return call_api(
                "stt",
                method="post",
                files={"audio": audio_file}
            )
    except Exception as e:
        return {"error": f"Lỗi khi xử lý file: {str(e)}"}


def update_model_info():
    """Cập nhật thông tin model trên giao diện"""
    info = get_model_info()
    if "error" in info:
        return info["error"]

    return f"""
    **Thông tin Model:**
    - Tên model: {info.get('model_name', 'N/A')}
    - Thư mục: `{info.get('models_dir', 'N/A')}`
    - Đường dẫn: `{info.get('model_path', 'N/A')}`
    """


def process_audio(audio_path: str):
    """Xử lý audio từ file path và gửi đến API"""
    if not audio_path:
        return "Vui lòng chọn file audio trước", ""

    try:
        response = requests.post(
            f"{API_URL}/stt",
            data={"audio_path": audio_path}  # Gửi dưới dạng form data
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                transcription = result.get("result", {})
                return transcription.get("text").strip()
        return "⚠️ Lỗi: " + response.json().get("detail", "Unknown error")

    except Exception as e:
        return f"⚠️ Lỗi kết nối: {str(e)}", ""



with gr.Blocks(title="Meeting Secretary") as demo:
    with gr.Sidebar(width=200):
        gr.Markdown("## Meeting Secretary")
        gr.Markdown("Upload or record audio to get transcription")

    with gr.Tab("Offline Meeting Secretary"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=['microphone', 'upload'],
                    type="filepath",
                    label="Audio Input",
                    interactive=True
                )
                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Transcription",
                    placeholder="Your transcription will appear here...",
                    lines=15,
                    interactive=True
                )

        submit_btn.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=output_text
        )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
