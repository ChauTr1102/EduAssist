
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class VinaLlamaSummarizer:
    def __init__(self, model_name="vilm/vinallama-7b", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32)
        self.model.to(self.device)
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device=="cuda" else -1
        )

    def summarize(self, text, max_length=128, min_length=30, do_sample=False):
        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample
        )
        return summary[0]["summary_text"]

if __name__ == "__main__":
    # Example usage
    text = """
    Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á, có lịch sử lâu đời và nền văn hóa phong phú. Trong những năm gần đây, Việt Nam đã đạt được nhiều thành tựu nổi bật về kinh tế, giáo dục và công nghệ, thu hút sự quan tâm của cộng đồng quốc tế.
    """
    summarizer = VinaLlamaSummarizer()
    summary = summarizer.summarize(text)
    print("Tóm tắt:", summary)
