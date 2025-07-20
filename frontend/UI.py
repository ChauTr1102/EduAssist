import gradio as gr
from faster_whisper import WhisperModel
import os
model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

def get_transcribe(audio):
    if not audio:
        return "No audio file provided"

    # Check if file exists
    if not os.path.exists(audio):
        return "Error: File not found"

    try:
        segments, info = model.transcribe(audio, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        full_text = ""
        for segment in segments:
            full_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

        return full_text.strip()

    except Exception as e:
        return f"Error during transcription: {str(e)}"


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

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
