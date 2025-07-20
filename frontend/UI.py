

import gradio as gr
from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")




def get_transcribe(audio):

    segments, info = model.transcribe(audio, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = ""
    for segment in segments:
        text +=  ("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)) +'/n'
    return text
with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Textbox('hello')

    with gr.Tab("Offline meeting secretary"):
        gr.Markdown("Your offline meeting secretary!")
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    audio_from_user = gr.Audio(
                        sources=['microphone', 'upload'],
                        type="filepath",  # Luôn trả về đường dẫn file
                        format="wav",  # Đảm bảo luôn là file WAV
                        label="Record or Upload Audio"
                    )
                    submit_button = gr.Button('Submit', variant="primary")

                with gr.Column(scale=4):
                    output_transcribe = gr.Textbox(
                        label="Transcription Result",
                        placeholder="Text will appear here..."
                    )

            submit_button.click(
                fn= get_transcribe,
                inputs=audio_from_user,
                outputs=output_transcribe
            )

# Uncomment để chạy demo
demo.launch()