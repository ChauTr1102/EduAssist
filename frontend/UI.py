import gradio as gr
import requests
import tempfile
import os
import random
from typing import Optional
import json

current_transcript = ""
current_video_path  = ''
file_type = " "
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
    global current_transcript
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
                current_transcript = result.get("result", {})
                current_transcript  =  current_transcript.get("transcribe_by_sentence").strip()
                return current_transcript
        return "⚠️ Lỗi: " + response.json().get("detail", "Unknown error")

    except Exception as e:
        return f"⚠️ Lỗi kết nối: {str(e)}", ""

def get_response_from_bot(message, history):
    global current_transcript
    prompt = f"""
        Hãy trả lời các câu hỏi mà tôi hỏi dựa vào thông tin của 
        đoạn tóm tắt này, hãy trả lời chính xác, nếu không biết hãy nói không biết
        và nêu như chưa có cuộc hội thoại hãy báo là bạn chưa có cụôc hội thoại nào 
        và cuộc thoại đó đây

        {current_transcript}

        câu hỏi đó là {message}
"""

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={  # Sửa thành json thay vì form data để gửi cấu trúc phức tạp
                "prompt": prompt,
                "model": "llama1",  # Có thể thay đổi model phù hợp cho summarization
                "stream": False
            },
            timeout=300
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


def summarization(transcript: str, api_url: str = None) -> str:
    global current_transcript
    """
    Tóm tắt nội dung hội thoại bằng AI

    Args:
        transcript (str): Nội dung hội thoại cần tóm tắt
        api_url (str, optional): URL endpoint API. Mặc định sẽ dùng biến toàn cục API_URL

    Returns:
        str: Nội dung đã được tóm tắt hoặc thông báo lỗi
    """
    # Định dạng prompt chuyên biệt cho tóm tắt hội thoại
    prompt = f"""
    Hãy tóm tắt cuộc hội thoại sau đây thành các ý chính ngắn gọn, 
    giữ nguyên các thông tin quan trọng như tên riêng, số liệu, 
    và các quyết định quan trọng:

    {current_transcript}

    Yêu cầu:
    - Ngôn ngữ giữ nguyên như bản gốc
    - Độ dài khoảng 20-30% so với bản gốc
    - Đánh dấu các điểm quan trọng bằng bullet points (•)
    """

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={  # Sửa thành json thay vì form data để gửi cấu trúc phức tạp
                "prompt": prompt,
                "model": "llama1",  # Có thể thay đổi model phù hợp cho summarization
                "stream": False
            },
            timeout=300
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


with gr.Blocks(title="Meeting Secretary",fill_height=True) as demo:
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

                submit_audio_btn = gr.Button("Transcribe", variant="primary")
                gr.ChatInterface(get_response_from_bot, type="messages", autofocus=False,fill_height=True,save_history=True)

            with gr.Column(scale=4):
                summarization_box = gr.Textbox(
                    label="Summarization",
                    placeholder="Your Summarization will appear here...",
                    lines=15,
                    interactive=True
                )
                video_input = gr.Video(
                    sources=["upload", "webcam"],
                    label="upload or capture video",
                    interactive=True,
                    height=245
                )
                submit_video_btn = gr.Button("Transcribe", variant="primary")


            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Transcription",
                    placeholder="Your transcription will appear here...",
                    lines=31,
                    interactive=True,
                )
        download_box = gr.Textbox(
            label="Downloading audio file",
            placeholder="Your file is store in here",
            interactive=True,
        )
    submit_audio_btn.click(
        fn=get_transcribe,
        inputs=audio_input,
        outputs=output_text
    ).then(
        fn = summarization,
        inputs = current_transcript,
        outputs = summarization_box
    )
    submit_video_btn.click(
        fn=extract_video_to_audio,
        inputs=video_input,
        outputs = download_box
    ).then(
        fn=get_transcribe,
        inputs=download_box,
        outputs = output_text
    ).then(
        fn = summarization,
         inputs =current_transcript,
        outputs = summarization_box
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
