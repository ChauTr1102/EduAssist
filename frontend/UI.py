import gradio as gr
import requests
import tempfile
import os
import random
from typing import Optional


current_transcript = ""

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

def extract_video_to_audio(video_path:str,output_dir: Optional[str] = None):
    global current_video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

        # Chuẩn bị payload
    payload = {
        "video_path": video_path
    }
    if output_dir:
        payload["output_dir"] = output_dir

    try:
        # Gọi API
        response = requests.post(
            f"{API_URL}/extract-audio",
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        if result["status"] != "success":
            raise Exception(f"API error: {result.get('detail', 'Unknown error')}")
        current_video_path = result["audio_path"]
        return result["audio_path"]

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def get_transcribe(audio_path: str):
    """Xử lý audio từ file path và gửi đến API"""
    if not audio_path:
        return "Vui lòng chọn file audio trước"

    try:
        response = requests.post(
            f"{API_URL}/stt",
            data={"audio_path": audio_path}
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                transcript = result.get("result", {}).get("transcribe_by_sentence", "").strip()
                return transcript
        return "⚠️ Lỗi: " + response.json().get("detail", "Unknown error")

    except Exception as e:
        return f"⚠️ Lỗi kết nối: {str(e)}"


def random_response(message, history):
    return random.choice(["Yes", "No"])

def summarization(transcript: str, user_prompt: str) -> str:
    """
    Tóm tắt nội dung hội thoại bằng AI

    Args:
        transcript (str): Nội dung hội thoại cần tóm tắt
        api_url (str, optional): URL endpoint API. Mặc định sẽ dùng biến toàn cục API_URL

    Returns:
        str: Nội dung đã được tóm tắt hoặc thông báo lỗi
    """
    # Định dạng prompt chuyên biệt cho tóm tắt hội thoại
    prompt = f"""Bạn sẽ nhận được một lời nhắc từ người dùng và một đoạn hội thoại được tách ra từ file audio, 
        hãy phân tích và tóm tắt lại nội dung có trong cuộc hội thoại đó theo yêu cầu của người dùng. Nếu không có yêu cầu nào từ người dùng,
        hãy tóm tắt lại theo yêu cầu của hệ thống.

        Lời nhắc của người dùng:
        {user_prompt}

        Chi tiết cuộc hội thoại:
        {transcript}
        """
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={  # Sửa thành json thay vì form data để gửi cấu trúc phức tạp
                "prompt": prompt,
                "model": "llama1",  # Có thể thay đổi model phù hợp cho summarization
                "stream": False
            },
            timeout=30
        )

        response.raise_for_status()  # Tự động raise exception nếu có lỗi HTTP

        result = response.json()

        # Xử lý response từ API
        if isinstance(result, dict):
            return result.get("response", "⚠️ Không nhận được nội dung tóm tắt")
        elif isinstance(result, str):
            return result
        else:
            return "⚠️ Định dạng response không hợp lệ"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Lỗi kết nối: {str(e)}"
    except ValueError as e:
        return f"⚠️ Lỗi xử lý dữ liệu: {str(e)}"
    except Exception as e:
        return f"⚠️ Lỗi không xác định: {str(e)}"


def summarization_gemini(transcript: str, user_prompt: str) -> str:
    """
    Tóm tắt nội dung hội thoại bằng AI

    Args:
        transcript (str): Nội dung hội thoại cần tóm tắt
        api_url (str, optional): URL endpoint API. Mặc định sẽ dùng biến toàn cục API_URL

    Returns:
        str: Nội dung đã được tóm tắt hoặc thông báo lỗi
    """
    # Định dạng prompt chuyên biệt cho tóm tắt hội thoại
    prompt = f"""Bạn sẽ nhận được một lời nhắc từ người dùng và một đoạn hội thoại được tách ra từ file audio, 
    hãy phân tích và tóm tắt lại nội dung có trong cuộc hội thoại đó theo yêu cầu của người dùng. Nếu không có yêu cầu nào từ người dùng,
    hãy tóm tắt lại theo yêu cầu của hệ thống.
    
    Lời nhắc của người dùng:
    {user_prompt}
    
    Chi tiết cuộc hội thoại:
    {transcript}
    """
    try:
        response = requests.post(
            f"{API_URL}/summarize_gemini",
            json={  # Sửa thành json thay vì form data để gửi cấu trúc phức tạp
                "script": prompt,
            },
            timeout=30
        )

        response.raise_for_status()  # Tự động raise exception nếu có lỗi HTTP

        result = response.json()

        # Xử lý response từ API
        if isinstance(result, dict):
            return result.get("response", "⚠️ Không nhận được nội dung tóm tắt")
        elif isinstance(result, str):
            return result
        else:
            return "⚠️ Định dạng response không hợp lệ"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Lỗi kết nối: {str(e)}"
    except ValueError as e:
        return f"⚠️ Lỗi xử lý dữ liệu: {str(e)}"
    except Exception as e:
        return f"⚠️ Lỗi không xác định: {str(e)}"


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

                gr.ChatInterface(random_response, type="messages", autofocus=False)

            with gr.Column(scale=4):
                prompt_box = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Edit your Prompt here",
                    lines=3,
                    interactive=True
                )

                summarization_box = gr.Textbox(
                    label="Summarization",
                    placeholder="Your Summarization will appear here...",
                    lines=15,
                    interactive=True
                )
                summarise_btn = gr.Button("Summarise Again", variant="secondary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Transcription",
                    placeholder="Your transcription will appear here...",
                    lines=15,
                    interactive=True
                )
    submit_btn.click(
        fn=get_transcribe,
        inputs=audio_input,
        outputs=output_text
    ).then(
        fn=summarization,
        inputs=[output_text, prompt_box],
        outputs=summarization_box
    )

    # Bấm Summarise Again
    summarise_btn.click(
        fn=summarization,
        inputs=[output_text, prompt_box],
        outputs=summarization_box
    )

    with gr.Tab("Online Meeting Secretary"):
        with gr.Row():
            with gr.Column(scale=1):
                on_audio_input = gr.Audio(
                    sources=['microphone', 'upload'],
                    type="filepath",
                    label="Audio Input",
                    interactive=True
                )
                on_submit_btn = gr.Button("Transcribe", variant="primary")

                gr.ChatInterface(random_response, type="messages", autofocus=False)

            with gr.Column(scale=4):
                on_prompt_box = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Edit your Prompt here",
                    lines=3,
                    interactive=True
                )

                on_summarization_box = gr.Textbox(
                    label="Summarization",
                    placeholder="Your Summarization will appear here...",
                    lines=15,
                    interactive=True
                )
                on_summarise_btn = gr.Button("Summarise Again", variant="secondary")

            with gr.Column(scale=2):
                on_output_text = gr.Textbox(
                    label="Transcription",
                    placeholder="Your transcription will appear here...",
                    lines=15,
                    interactive=True
                )

    on_submit_btn.click(
        fn=get_transcribe,
        inputs=on_audio_input,
        outputs=on_output_text
    ).then(
        fn=summarization_gemini,
        inputs=[on_output_text, on_prompt_box],
        outputs=on_summarization_box
    )

    # Bấm Summarise Again
    on_summarise_btn.click(
        fn=summarization_gemini,
        inputs=[on_output_text, on_prompt_box],
        outputs=on_summarization_box
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)