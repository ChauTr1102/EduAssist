"""
Ví dụ sử dụng MapReduce với model custom Qw3:4b_4bit
Model: shmily_006/Qw3:4b_4bit (finetune từ Qwen3 4B)
"""

import asyncio
import sys

# Thêm path để import
sys.path.insert(0, '/home/bojjoo/Code/EduAssist')

from api.services.local_llm_mapreduce import LanguageModelOllamaMapReduce, OllamaMapReduceLLM

ollama_model = LanguageModelOllamaMapReduce(
        model="shmily_006/Qw3:4b_4bit",  # Model custom của bạn
        tokenizer_name="Qwen/Qwen3-4B",
        temperature=0.5
    )

mapreduce_llm = OllamaMapReduceLLM(
        model=ollama_model,
        context_window=4096,  # Điều chỉnh theo model của bạn
        collapse_threshold=2048
    )

with open("/home/bojjoo/Code/EduAssist/test_data/dienvien Cong Ly.txt") as f:
    long_document = f.read()

result = mapreduce_llm.process_long_text(long_document, "Tóm tắt các ý chính")
print(result["answer"])
