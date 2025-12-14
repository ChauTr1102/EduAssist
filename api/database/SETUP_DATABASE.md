# EduAssist Database Setup - Quick Start Guide

## 📋 Tổng quan

Database PostgreSQL cho hệ thống EduAssist Meeting Assistant với các tính năng:
- Quản lý meetings (cuộc họp)
- Upload và embedding documents (tài liệu)
- Lưu transcript (bản ghi âm)
- Q&A conversation (hội thoại)
- Tạo summaries (tóm tắt)

## 🚀 Quick Start (5 phút)

### Bước 1: Cài đặt PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Kiểm tra PostgreSQL đang chạy
sudo systemctl status postgresql
```

### Bước 2: Tạo Database

```bash
# Đăng nhập PostgreSQL
sudo -u postgres psql

# Trong psql, chạy:
CREATE DATABASE eduassist;
\q
```

### Bước 3: Cấu hình môi trường

Tạo file `.env` trong root project:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eduassist
```

### Bước 4: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Các package chính đã được thêm:
- `SQLAlchemy==2.0.44`
- `psycopg2-binary==2.9.10`  
- `alembic==1.14.0`

### Bước 5: Khởi tạo tables

```bash
python scripts/init_database.py
```

Chọn option **1** để tạo tables.

✅ Done! Database đã sẵn sàng.

## 📂 Cấu trúc Files

```
api/
├── database/
│   ├── __init__.py          # Package init
│   ├── models.py            # SQLAlchemy models (Meeting, Document, etc.)
│   ├── database.py          # DB connection & session
│   ├── crud.py              # CRUD operations
│   └── README.md            # Chi tiết documentation
├── services/
│   └── meeting_service.py   # Service layer (high-level API)
scripts/
└── init_database.py         # Database initialization script
```

## 🎯 Cách sử dụng

### Option 1: Sử dụng Service Layer (Recommended)

```python
from api.services.meeting_service import meeting_service

# 1. Tạo meeting
meeting_id = meeting_service.create_new_meeting(
    title="Meeting Q1 2025",
    description="Họp kế hoạch Q1"
)

# 2. Upload document
doc_id = meeting_service.upload_document(
    meeting_id=meeting_id,
    file_path="/path/to/document.pdf"
)

# 3. Embed document
embed_info = meeting_service.embed_document(meeting_id, doc_id)
print(f"Embedded {embed_info['chunk_count']} chunks")

# 4. Generate meeting context
context = meeting_service.generate_meeting_context(meeting_id, top_k=10)

# 5. Save transcript
transcript_id = meeting_service.save_transcript(
    meeting_id=meeting_id,
    content="Transcript content...",
    duration_ms=1800000
)

# 6. Q&A
meeting_service.ask_question(
    meeting_id=meeting_id,
    question="Câu hỏi?",
    answer="Câu trả lời...",
    metadata={"sources": ["doc.pdf"]}
)

# 7. Get conversation history
history = meeting_service.get_conversation_history(meeting_id)

# 8. Create summary
summary_id = meeting_service.create_summary(
    meeting_id=meeting_id,
    content="Summary content...",
    summary_type="general"
)

# 9. Get full meeting info
info = meeting_service.get_meeting_info(meeting_id)
```

### Option 2: Sử dụng CRUD trực tiếp

```python
from api.database.database import SessionLocal
from api.database import crud

db = SessionLocal()

meeting = crud.create_meeting(db, title="Demo Meeting")
doc = crud.create_document(db, meeting.id, "file.pdf", "/path/to/file.pdf", "pdf")
# ... more operations

db.close()
```

## 🔧 Tích hợp vào UI Flow

### Flow như NotebookLM:

```python
from api.services.meeting_service import meeting_service

# === USER CREATES NEW MEETING ===
meeting_id = meeting_service.create_new_meeting(
    title=user_input_title,
    description=user_input_description
)

# === USER UPLOADS DOCUMENTS ===
for uploaded_file in user_uploaded_files:
    doc_id = meeting_service.upload_document(meeting_id, uploaded_file.path)
    
    # Embed document in background
    embed_info = meeting_service.embed_document(meeting_id, doc_id)

# === GENERATE MEETING CONTEXT ===
meeting_context = meeting_service.generate_meeting_context(meeting_id, top_k=10)

# === START RECORDING (từ new_ui.py) ===
# ... user clicks "Start Recording"
# ... ASR processing ...
transcript_content = asr_output

# Save transcript
meeting_service.save_transcript(
    meeting_id=meeting_id,
    content=transcript_content,
    duration_ms=recording_duration
)

# === USER ASKS QUESTIONS ===
def on_user_question(question):
    # Get conversation history
    history = meeting_service.get_conversation_history(meeting_id)
    
    # RAG processing with context + history
    answer = rag_pipeline(question, meeting_context, history)
    
    # Save to database
    meeting_service.ask_question(
        meeting_id=meeting_id,
        question=question,
        answer=answer,
        metadata={"sources": retrieved_docs}
    )
    
    return answer

# === GENERATE SUMMARY ===
summary = llm_generate_summary(transcript_content, meeting_context)
meeting_service.create_summary(
    meeting_id=meeting_id,
    content=summary,
    summary_type="general",
    title="Tóm tắt cuộc họp"
)
```

## 📊 Database Schema

```
Meeting (1) ─── (1) Transcript
   │
   ├─── (1) Conversation ─── (*) Message  
   │
   ├─── (*) Summarize
   │
   └─── (*) Document
```

Chi tiết các bảng xem trong `api/database/README.md`

## 🛠️ Troubleshooting

### "Connection refused"
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Auto-start on boot
```

### "Password authentication failed"
Sửa `.env`:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/eduassist
```

### "Tables already exist"
```bash
python scripts/init_database.py
# Chọn option 3 (Reset database)
```

### Xem logs PostgreSQL
```bash
sudo tail -f /var/log/postgresql/postgresql-*.log
```

## 📚 Tài liệu chi tiết

- Database models & schema: `api/database/README.md`
- CRUD operations: Xem `api/database/crud.py`
- Service API: Xem `api/services/meeting_service.py`
- Complete example: `examples/complete_flow_example.py`