import gradio as gr
from faster_whisper import WhisperModel
import os
# Use the pipeline directly for summarization
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from phobert_ollama_text_summarization import VietnameseSummarizationPipeline
# model_size = "large-v3"

# # Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")


# Initialize the pipeline once
pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = VietnameseSummarizationPipeline()
        except Exception as e:
            return None, f"Error initializing pipeline: {str(e)}"
    return pipeline, None

def summarize_text(text, summary_length):
    if not text or len(text.strip()) < 10:
        return "Please enter at least 10 characters of Vietnamese text."
    pipe, err = get_pipeline()
    if err:
        return err
    try:
        results = pipe.process(text.strip(), summary_length)
        return results.get('vietnamese_summary', 'No summary returned.')
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def get_transcribe(audio):
    if not audio:
        return "No audio file provided"

    # Check if file exists
    if not os.path.exists(audio):
        return "Error: File not found"

    # try:
    #     segments, info = model.transcribe(audio, beam_size=5)
    #     print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    #     full_text = ""
    #     for segment in segments:
    #         full_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

    #     return full_text.strip()

    # except Exception as e:
    #     return f"Error during transcription: {str(e)}"


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
            fn=get_transcribe,
            inputs=audio_input,
            outputs=output_text
        )

    with gr.Tab("Vietnamese Text Summarization"):
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Vietnamese Text",
                    placeholder="Nhập văn bản tiếng Việt để tóm tắt...",
                    lines=8,
                    interactive=True
                )
                summary_length = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=50,
                    step=1,
                    label="Summary Length (words)"
                )
                summarize_btn = gr.Button("Summarize", variant="primary")

            with gr.Column(scale=2):
                summary_output = gr.Textbox(
                    label="Summary",
                    placeholder="Bản tóm tắt sẽ xuất hiện ở đây...",
                    lines=10,
                    interactive=True
                )

        summarize_btn.click(
            fn=summarize_text,
            inputs=[input_text, summary_length],
            outputs=summary_output
        )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
