# Hướng dẫn sử dụng MapReduce với Ollama

## Tổng quan

MapReduce LLM framework cho phép xử lý các văn bản dài vượt quá giới hạn context window của model bằng cách:
1. **Map**: Chia văn bản thành các chunks nhỏ và xử lý từng chunk
2. **Collapse**: Gộp các kết quả lại
3. **Reduce**: Tổng hợp thành câu trả lời cuối cùng

## Cài đặt

### 1. Cài đặt dependencies

```bash
# Cài đặt transformers cho tokenizer
pip install transformers torch

# Cài đặt LLM_MapReduce package
cd /home/bojjoo/Code/LLM_MapReduce
pip install -e .
```

### 2. Đảm bảo Ollama đang chạy

```bash
# Kiểm tra Ollama
curl http://localhost:11434/api/tags

# Pull model nếu chưa có
ollama pull llama3.2:3b
```

## Cách sử dụng

### Phương án 1: Sử dụng class mở rộng (Khuyến nghị)

Class `LanguageModelOllamaMapReduce` đã tích hợp sẵn tokenizer và format output.

```python
import asyncio

from api.services.local_llm_mapreduce import LanguageModelOllamaMapReduce, OllamaMapReduceLLM

async def main():
    # CÁCH 1: Tự động detect tokenizer (KHUYẾN NGHỊ)
    # Class sẽ tự động chọn tokenizer phù hợp dựa trên tên model
    ollama_model = LanguageModelOllamaMapReduce(
        model="shmily_006/Qw3:4b_4bit",  # Model finetune của Qwen3
        # tokenizer_name=None (mặc định) → tự động chọn Qwen2.5 tokenizer
        temperature=0.7
    )
    
    # CÁCH 2: Chỉ định tokenizer thủ công
    # ollama_model = LanguageModelOllamaMapReduce(
    #     model="shmily_006/Qw3:4b_4bit",
    #     tokenizer_name="Qwen/Qwen2.5-7B",  # Chỉ định rõ tokenizer
    #     temperature=0.7
    # )
    
    # Khởi tạo OllamaMapReduceLLM (custom implementation cho Ollama)
    mapreduce_llm = OllamaMapReduceLLM(
        model=ollama_model,
        context_window=2048,
        collapse_threshold=1024
    )
    
    # Văn bản dài
    document = "... văn bản rất dài ..."
    query = "Tóm tắt nội dung chính"
    
    # Xử lý
    result = mapreduce_llm.process_long_text(document, query)
    print(result["answer"])
    
    await ollama_model.aclose()

asyncio.run(main())
```

### Phương án 2: Sử dụng Wrapper (Không khuyến nghị với Ollama)

**Lưu ý:** Do Ollama API khác với PyTorch models, bạn nên dùng `OllamaMapReduceLLM` 
thay vì `MapReduceLLM` từ package gốc.

Xem file `example_qw3_mapreduce.py` để biết cách sử dụng chi tiết.

## Tham số quan trọng

### LanguageModelOllamaMapReduce

- **model**: Tên model đã pull trong Ollama 
  - Models chính thức: `"llama3.2:3b"`, `"qwen2.5:7b"`, `"gemma2:2b"`
  - Models finetune/custom: `"shmily_006/Qw3:4b_4bit"`, `"your_model:tag"`
  
- **tokenizer_name**: (Optional) Tokenizer từ HuggingFace
  - Nếu `None` (mặc định): **Tự động detect** dựa trên tên model
  - Tự động detect hỗ trợ: Qwen/Qw, Llama, Phi, Gemma, Mistral
  - Có thể chỉ định thủ công nếu muốn:
    ```python
    # Ví dụ: Model finetune từ Qwen nhưng tên không chứa "qwen"
    ollama_model = LanguageModelOllamaMapReduce(
        model="my_custom_model:1b",
        tokenizer_name="Qwen/Qwen2.5-7B"  # Chỉ định rõ
    )
    ```

- **temperature**: Độ sáng tạo (0.0-1.0), khuyến nghị 0.7
- **host**: Địa chỉ Ollama server, mặc định "http://localhost:11434"

### Auto-detect Tokenizer Logic

Class sẽ tự động chọn tokenizer dựa trên tên model:

| Tên model chứa | Tokenizer được chọn | Ví dụ |
|----------------|---------------------|-------|
| `qwen`, `qw` | `Qwen/Qwen2.5-7B` | `qwen2.5:7b`, `Qw3:4b`, `shmily_006/Qw3:4b_4bit` |
| `llama` | `meta-llama/Llama-3.2-3B` | `llama3.2:3b`, `llama3.1:8b` |
| `phi` | `microsoft/Phi-3-mini-4k-instruct` | `phi3:mini` |
| `gemma` | `google/gemma-2-2b` | `gemma2:2b` |
| `mistral`, `mixtral` | `mistralai/Mistral-7B-v0.1` | `mistral:7b` |
| Khác | `vinai/phobert-base` | Mặc định (tiếng Việt) |

### MapReduceLLM

- **context_window**: Kích thước context window của model
  - llama3.2:3b: 2048-4096
  - Các model lớn hơn: 4096-8192
- **collapse_threshold**: Ngưỡng để gộp chunks, thường là 1/2 của context_window

## Ví dụ thực tế

### Xử lý biên bản cuộc họp dài

```python
from api.services.local_llm_mapreduce import LanguageModelOllamaMapReduce
from llm_mapreduce.mapreduce import MapReduceLLM

# Đọc biên bản
with open("bien_ban_cuoc_hop.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

# Khởi tạo
ollama = LanguageModelOllamaMapReduce(
    model="qwen2.5:7b",
    tokenizer_name="vinai/phobert-base"
)

mapreduce = MapReduceLLM(
    model=ollama,
    context_window=4096,
    collapse_threshold=2048
)

# Các loại query có thể dùng
queries = [
    "Tóm tắt các quyết định chính trong cuộc họp",
    "Liệt kê các vấn đề được thảo luận",
    "Tóm tắt ý kiến của từng thành viên"
]

for query in queries:
    result = mapreduce.process_long_text(transcript, query)
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    print(result["answer"])
```

### Tích hợp vào RAG pipeline

```python
async def process_long_document_with_mapreduce(
    document: str,
    question: str,
    ollama_model: LanguageModelOllamaMapReduce
):
    """
    Xử lý tài liệu dài trước khi đưa vào RAG
    """
    mapreduce = MapReduceLLM(
        model=ollama_model,
        context_window=4096
    )
    
    # Tóm tắt tài liệu dài trước
    summary_query = f"Tóm tắt nội dung liên quan đến: {question}"
    result = mapreduce.process_long_text(document, summary_query)
    
    # Sử dụng kết quả tóm tắt trong RAG
    return result["answer"]
```

## Lưu ý

1. **Tokenizer vs Model - Tại sao khác nhau?**
   
   **Vấn đề**: Ollama không cung cấp API để tokenize/đếm tokens
   
   **Giải pháp**: Dùng tokenizer tương tự từ HuggingFace để **ước lượng** số tokens
   
   **Tại sao được phép?**
   - Tokenizer chỉ dùng để chia chunk (preprocessing), không ảnh hưởng generation
   - Ollama vẫn dùng tokenizer gốc khi generate
   - Sai số 20-30% có thể chấp nhận được vì MapReduce có buffer
   
   **Chọn tokenizer phù hợp:**
   - ✅ Tiếng Việt: `"vinai/phobert-base"` (tốt nhất cho tiếng Việt)
   - ✅ Tiếng Anh: `"gpt2"` hoặc `"distilgpt2"`
   - ⚠️ Tránh: Dùng tokenizer tiếng Anh cho text tiếng Việt (sai số lớn)
   
   **Tokenizer lý tưởng cho từng model Ollama:**
   - Llama 3.x: Dùng `"meta-llama/Meta-Llama-3-8B"` nếu có quyền truy cập
   - Qwen 2.5: Dùng `"Qwen/Qwen2.5-7B"` 
   - Gemma: Dùng `"google/gemma-2b"`
   - Nếu không có: Dùng PhoBERT (tiếng Việt) hoặc GPT-2 (tiếng Anh)

2. **Context Window**: Kiểm tra context window thực tế của model
   ```bash
   ollama show llama3.2:3b --modelfile | grep num_ctx
   ```

3. **Memory**: MapReduce sẽ gọi model nhiều lần, cần đủ RAM

4. **Async vs Sync**: 
   - MapReduce hiện tại chỉ hỗ trợ sync (`generate()`)
   - Dùng `asyncio.run()` để wrap nếu cần async context

## Troubleshooting

### Lỗi "tokenizer not found"
```python
# Tải tokenizer thủ công
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
```

### Lỗi "Connection refused"
```bash
# Kiểm tra Ollama có đang chạy
systemctl status ollama
# hoặc
ollama serve
```

### Output không đúng format
```python
# Kiểm tra output của generate()
result = ollama_model.generate("test")
print(type(result), result)
# Phải là dict với key "text", "answer"
```

## Files liên quan

- `/home/bojjoo/Code/EduAssist/api/services/local_llm_mapreduce.py`: Class mở rộng
- `/home/bojjoo/Code/EduAssist/api/services/ollama_mapreduce_example.py`: Wrapper và examples
- `/home/bojjoo/Code/LLM_MapReduce/llm_mapreduce/mapreduce.py`: MapReduce core logic
