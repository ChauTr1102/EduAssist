from api.services import *

class FasterWhisper:
    def __init__(self, model_name):
        self.model = WhisperModel(model_name, device="cuda", compute_type="int8")

    def extract_text(self, audio):
        segments, info = self.model.transcribe(audio)
        text = " ".join([segment.text for segment in segments])
        return text