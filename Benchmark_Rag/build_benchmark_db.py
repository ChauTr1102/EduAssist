import json
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

# --- CẤU HÌNH ---
KB_PATH = "knowledge_base.jsonl"  # File dữ liệu gốc bạn vừa tải lên
DB_SAVE_PATH = "../vectorstores/Benchmark_rag"  # Nơi sẽ lưu Vector DB mới
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"


def build_db():
    print("1. Đang load model embedding (có thể mất chút thời gian lần đầu)...")
    model_embedding = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"trust_remote_code": True}
    )

    print(f"2. Đang đọc dữ liệu từ {KB_PATH}...")
    documents = []

    # Đọc file jsonl
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Docs"):
            try:
                item = json.loads(line)
                text = item.get("text", "")
                doc_id = item.get("doc_id", "")

                if text and doc_id:
                    # Tạo đối tượng Document của LangChain
                    # QUAN TRỌNG: Phải lưu doc_id vào metadata để sau này tính Recall
                    doc = Document(
                        page_content=text,
                        metadata={"doc_id": doc_id}
                    )
                    documents.append(doc)
            except Exception as e:
                continue

    print(f"   => Đã load được {len(documents)} tài liệu gốc.")

    print("3. Đang cắt nhỏ văn bản (Chunking)...")
    # Cấu hình giống hệt hệ thống thật của bạn để kết quả công bằng
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Cấu hình cho Documents
        chunk_overlap=200,
        length_function=len
    )

    chunked_docs = text_splitter.split_documents(documents)
    print(f"   => Sau khi cắt, có tổng cộng {len(chunked_docs)} chunks.")

    print("4. Đang tạo Vector Index (FAISS)... (Bước này tốn GPU/CPU nhất)")
    # Batch size nhỏ để tránh tràn RAM nếu máy yếu, nhưng FAISS xử lý khá tốt
    # LangChain tự động batching, nhưng ta cứ đưa hết vào
    db = FAISS.from_documents(chunked_docs, model_embedding)

    print(f"5. Đang lưu DB xuống thư mục: {DB_SAVE_PATH}")
    db.save_local(DB_SAVE_PATH)
    print("✅ HOÀN TẤT! Đã tạo xong dữ liệu để benchmark.")


if __name__ == "__main__":
    if not os.path.exists(KB_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {KB_PATH}. Hãy đảm bảo file này nằm cùng thư mục.")
    else:
        build_db()