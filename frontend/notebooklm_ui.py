"""
Meeting Assistant UI - NotebookLM Style
Clean, modern interface with smooth workflow
"""

import os
import sys
import threading
from typing import Optional
import asyncio
from queue import Queue, Empty
import time
import warnings

import gradio as gr
import pynini
from punctuators.models import PunctCapSegModelONNX
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.chunkformer_stt import ChunkFormer
from api.private_config import *
from api.config import *
from api.services.vcdb_faiss import VectorStore
from api.services.local_llm import LanguageModelOllama
from api.services.rag_processor import RagProcessor
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from api.database.database import SessionLocal
from api.database import crud
from llm_mapreduce.ollama_mapreduce import OllamaMapReducePipeline, load_config


warnings.filterwarnings("ignore")

# =========================
# GLOBAL STATE
# =========================
current_meeting_id: Optional[int] = None
current_meeting_title: str = ""
current_meeting_context: str = ""

meeting_faiss: Optional[VectorStore] = None
transcript_faiss: Optional[VectorStore] = None
cache_faiss: Optional[VectorStore] = None

# =========================
# ITN MODEL
# =========================
def init_itn_model(itn_model_dir: str):
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")
    
    reader_classifier = pynini.Far(classifier_far)
    reader_verbalizer = pynini.Far(verbalizer_far)
    return reader_classifier.get_fst(), reader_verbalizer.get_fst()

# =========================
# INIT MODELS
# =========================
chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
punc_model = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ort_providers=["CPUExecutionProvider"],
)
itn_classifier, itn_verbalizer = init_itn_model(ITN_REPO)
llm = LanguageModelOllama("shmily_006/Qw3:4b_4bit", temperature=0.5)
model_embedding = HuggingFaceEmbeddings(
    model_name=MODEL_EMBEDDING,
    model_kwargs={"trust_remote_code": True}
)

config = load_config("/home/bojjoo/Code/EduAssist/api/services/config_ollama_mapreduce.yaml")
mapreduce_pipeline = OllamaMapReducePipeline(config)

# =========================
# QUEUES & LOCKS
# =========================
job_queue = Queue(maxsize=0)
summarizer_queue = Queue(maxsize=0)
embedding_queue = Queue(maxsize=0)

faiss_lock = threading.Lock()
transcript_lock = threading.Lock()
summary_lock = threading.Lock()
db_lock = threading.Lock()

# =========================
# ASYNC LOOP
# =========================
_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None

def start_async_loop():
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None:
        _ASYNC_LOOP = asyncio.new_event_loop()
        t = threading.Thread(target=lambda: asyncio.set_event_loop(_ASYNC_LOOP) or _ASYNC_LOOP.run_forever(), daemon=True)
        t.start()

def run_async(coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result(timeout=timeout)

start_async_loop()

# =========================
# UI STATE
# =========================
asr_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
transcript_text = ""
summary_text = ""

rag_processor = RagProcessor(
    job_queue=job_queue,
    embedding_queue=embedding_queue,
    n_commits_to_combine=3,
    overlap_m=1,
    timeout_sec=15.0,
)

# =========================
# WORKERS
# =========================
def worker_loop(worker_id: int):
    while True:
        try:
            text = job_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            normalize_prompt = llm.normalize_text(current_meeting_context, text)
            normalized = run_async(llm.async_generate(normalize_prompt), timeout=60.0)
            
            if not normalized or normalized.strip().casefold() == "none":
                continue
            
            if cache_faiss and cache_faiss.is_already_retrieved(normalized, similarity_threshold=0.7):
                continue
            
            related_docs = ""
            if meeting_faiss:
                related_docs = run_async(meeting_faiss.hybrid_search(normalized), timeout=60.0)
            
            summarizer_queue.put({"utterance": normalized, "related_docs": related_docs})
            
            if cache_faiss:
                with faiss_lock:
                    cache_faiss.add_cache(normalized)
        except Exception as e:
            print(f"[Worker-{worker_id}] ERROR: {e}")
        finally:
            job_queue.task_done()

def embedding_worker():
    while True:
        try:
            item = embedding_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            if isinstance(item, dict):
                clean_text = (item.get("text") or "").strip()
                start_ms = item.get("start_time_ms", 0)
                end_ms = item.get("end_time_ms", 0)
            else:
                clean_text = (item or "").strip()
                start_ms = end_ms = 0
            
            if clean_text and transcript_faiss:
                with faiss_lock:
                    transcript_faiss.add_transcript(clean_text, start_ms, end_ms)
        except Exception as e:
            print(f"[EmbeddingWorker] ERROR: {e}")
        finally:
            try:
                embedding_queue.task_done()
            except:
                pass

def summarizer_loop():
    global summary_text
    while True:
        try:
            item = summarizer_queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            utter = item.get("utterance", "")
            docs = item.get("related_docs", "")
            
            sum_prompt = SUMMARIZE_DOCUMENT_PROMPT.format(utterance=utter, related_docs=docs)
            summary = run_async(llm.async_generate(sum_prompt), timeout=60.0)
            
            with summary_lock:
                if summary_text:
                    summary_text = f"{summary_text}\n\n──────────\n{summary.strip()}"
                else:
                    summary_text = summary.strip()
        except Exception as e:
            print(f"[Summarizer] ERROR: {e}")
        finally:
            try:
                summarizer_queue.task_done()
            except:
                pass

# Start workers
for i in range(2):
    threading.Thread(target=worker_loop, args=(i+1,), daemon=True).start()
threading.Thread(target=summarizer_loop, daemon=True).start()
threading.Thread(target=embedding_worker, daemon=True).start()

# =========================
# CHUNKFORMER CALLBACK
# =========================
def on_update(event: str, payload: dict):
    global transcript_text
    
    with transcript_lock:
        if event == "partial":
            display = (payload.get("display") or "").strip()
            if display:
                transcript_text = display
        
        elif event == "commit":
            display = (payload.get("display") or payload.get("committed") or "").strip()
            if display:
                transcript_text = display
            
            new_commit = (payload.get("new_commit") or "").strip()
            if new_commit:
                rag_processor.process_new_commit(payload)
        
        elif event == "final_flush":
            text = (payload.get("text") or "").strip()
            if text:
                transcript_text = text
            rag_processor.flush_all(reason="final_flush")

# =========================
# ASR WORKER
# =========================
def asr_worker():
    try:
        chunkformer.chunkformer_asr_realtime_punc_norm(
            mic_sr=16000, stream_chunk_sec=0.5, lookahead_sec=0.5,
            left_context_size=128, right_context_size=32, max_overlap_match=32,
            vad_threshold=0.01, vad_min_silence_blocks=2,
            punc_model=punc_model, punc_window_words=100, punc_commit_margin_words=50,
            itn_classifier=itn_classifier, itn_verbalizer=itn_verbalizer,
            on_update=on_update, stop_event=stop_event, return_final=False,
        )
    except Exception as e:
        print(f"[ASR] Error: {e}")

# =========================
# DATABASE FUNCTIONS
# =========================
def create_meeting(title: str, description: str):
    global current_meeting_id, current_meeting_title, current_meeting_context
    global meeting_faiss, transcript_faiss, cache_faiss
    
    if not title.strip():
        return "⚠️ Vui lòng nhập tiêu đề cuộc họp", gr.update(visible=False)
    
    try:
        db = SessionLocal()
        try:
            meeting = crud.create_meeting(db, title=title, description=description)
            current_meeting_id = meeting.id
            current_meeting_title = meeting.title
            current_meeting_context = ""
            
            folder = f"meeting_{meeting.id}"
            meeting_faiss = VectorStore(folder+"/documents", model_embedding)
            transcript_faiss = VectorStore(folder+"/transcripts", model_embedding)
            cache_faiss = VectorStore(folder+"/cache", model_embedding)
            
            msg = f"""
### ✅ Cuộc họp đã được tạo!

**{title}**  
ID: `{meeting.id}` | Status: `{meeting.status}`

{description if description else '_Không có mô tả_'}

---
📎 **Bước tiếp theo:** Upload tài liệu liên quan đến cuộc họp
"""
            return msg, gr.update(visible=True)
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", gr.update(visible=False)

def upload_documents(files):
    global current_meeting_context
    
    if current_meeting_id is None:
        return "⚠️ Vui lòng tạo cuộc họp trước!", ""
    
    if not files:
        return "⚠️ Vui lòng chọn ít nhất một tài liệu!", ""
    
    try:
        db = SessionLocal()
        try:
            all_chunks = []
            doc_names = []
            
            for file in files:
                filename = os.path.basename(file.name)
                file_type = os.path.splitext(filename)[1].lower().replace('.', '')
                file_size = os.path.getsize(file.name) if os.path.exists(file.name) else 0
                
                doc = crud.create_document(
                    db=db, meeting_id=current_meeting_id,
                    filename=filename, file_path=file.name,
                    file_type=file_type, file_size=file_size
                )
                doc_names.append(filename)
                
                chunks = meeting_faiss.recursive_chunking(file.name)
                all_chunks.extend(chunks)
                
                crud.update_document_embedding(
                    db=db, document_id=doc.id,
                    vector_store_path=f"./vectorstores/meeting_{current_meeting_id}/documents",
                    embedding_model=MODEL_EMBEDDING, chunk_count=len(chunks)
                )
            
            if all_chunks:
                faiss_db = meeting_faiss.create_vectorstore(all_chunks)
                meeting_faiss.faiss_save_local(faiss_db, "")
                meeting_faiss.db = faiss_db

                documents = ""
                for i in chunks[:20]:
                    documents += i.page_content + "\n-----\n"


                question = "Tài liệu này nói về vấn đề gì, hãy trả lời theo format 'Meeting Context:'"
                current_meeting_context = mapreduce_pipeline.run(documents, question, chunk_size=4096)

                crud.update_meeting(
                    db=db, meeting_id=current_meeting_id,
                    meeting_context=current_meeting_context
                )
            
            msg = f"""
### ✅ Tài liệu đã được xử lý!

**Đã upload:** {len(doc_names)} tài liệu  
**Chunks:** {len(all_chunks)} đoạn văn bản  

📄 Files:
{chr(10).join([f'- {name}' for name in doc_names])}

---
🎤 **Bước tiếp theo:** Chuyển sang tab "Ghi âm" để bắt đầu cuộc họp
"""
            preview = current_meeting_context
            return msg, preview
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", ""

def start_recording():
    global asr_thread, transcript_text, summary_text
    
    if current_meeting_id is None:
        return gr.update(), gr.update(), "⚠️ Chưa tạo cuộc họp!"
    
    try:
        db = SessionLocal()
        try:
            crud.update_meeting(db, current_meeting_id, status="in_progress")
        finally:
            db.close()
    except:
        pass
    
    with transcript_lock:
        transcript_text = ""
    with summary_lock:
        summary_text = ""
    
    stop_event.clear()
    
    if asr_thread is None or not asr_thread.is_alive():
        asr_thread = threading.Thread(target=asr_worker, daemon=True)
        asr_thread.start()
        return gr.update(value=""), gr.update(value=""), "🎙️ Đang ghi âm..."
    else:
        return gr.update(), gr.update(), "✅ Đang ghi âm"

def stop_recording():
    global transcript_text
    
    if current_meeting_id is None:
        return "⚠️ Chưa có cuộc họp!"
    
    stop_event.set()
    rag_processor.flush_all(reason="stop")
    
    try:
        db = SessionLocal()
        try:
            with transcript_lock:
                final = transcript_text
            
            if final.strip():
                crud.create_transcript(
                    db=db, meeting_id=current_meeting_id,
                    content=final, duration_ms=0, language="vi"
                )
            
            crud.update_meeting(db, current_meeting_id, status="completed")
        finally:
            db.close()
        
        return "⏹️ Đã dừng và lưu transcript"
    except Exception as e:
        return f"⚠️ Lỗi: {e}"

def poll_ui():
    with transcript_lock:
        txt = transcript_text
    with summary_lock:
        sumtxt = summary_text
    return gr.update(value=txt), gr.update(value=sumtxt)

def chat_qa(history, message):
    if current_meeting_id is None:
        return (history or []) + [(message, "⚠️ Vui lòng tạo cuộc họp trước!")], ""
    
    if not message:
        return history, ""
    
    try:
        db = SessionLocal()
        try:
            db_history = crud.get_conversation_history(db, current_meeting_id, last_n=5)
            history_str = "\n\n".join([
                f"{'User' if h['role']=='human' else 'AI'}: {h['content']}" 
                for h in db_history
            ])
            
            reformulated = run_async(
                llm.reformulate_question(message, history_str, current_meeting_context),
                timeout=60.0
            )
            
            if reformulated.get("type") == 0:
                reply = run_async(
                    llm.normal_qa_handler(
                        reformulated["new_question"],
                        history_str, current_meeting_context
                    ), timeout=60.0
                )
            else:
                related_docs = ""
                related_transcript = ""
                
                if meeting_faiss:
                    related_docs = run_async(
                        meeting_faiss.hybrid_search(reformulated["new_question"]),
                        timeout=60.0
                    )
                
                if transcript_faiss and transcript_faiss.db:
                    related_transcript = run_async(
                        transcript_faiss.hybrid_search(reformulated["new_question"]),
                        timeout=60.0
                    )
                
                reply = run_async(
                    llm.rag_qa_handler(
                        reformulated["new_question"], history_str,
                        current_meeting_context, related_docs, related_transcript
                    ), timeout=60.0
                )
            
            crud.add_message(db, current_meeting_id, role="human", content=message)
            crud.add_message(db, current_meeting_id, role="ai", content=reply,
                           extra_data={"sources": ["documents", "transcripts"]})
        finally:
            db.close()
        
        return (history or []) + [(message, reply)], ""
    except Exception as e:
        return (history or []) + [(message, f"❌ Lỗi: {e}")], ""

def load_meetings():
    try:
        db = SessionLocal()
        try:
            meetings = crud.get_all_meetings(db, skip=0, limit=50)
            choices = [(f"{m.title} (ID: {m.id})", m.id) for m in meetings]
            return gr.update(choices=choices)
        finally:
            db.close()
    except:
        return gr.update(choices=[])

def select_meeting(meeting_id):
    global current_meeting_id, current_meeting_title, current_meeting_context
    global meeting_faiss, transcript_faiss, cache_faiss
    
    if not meeting_id:
        return "⚠️ Vui lòng chọn cuộc họp!", ""
    
    try:
        db = SessionLocal()
        try:
            meeting = crud.get_meeting(db, meeting_id)
            if not meeting:
                return "❌ Không tìm thấy cuộc họp!", ""
            
            current_meeting_id = meeting.id
            current_meeting_title = meeting.title
            current_meeting_context = meeting.meeting_context or ""
            
            folder = f"meeting_{meeting.id}"
            meeting_faiss = VectorStore(folder+"/documents", model_embedding)
            transcript_faiss = VectorStore(folder+"/transcripts", model_embedding)
            cache_faiss = VectorStore(folder+"/cache", model_embedding)
            
            docs = crud.get_documents(db, meeting_id)
            transcript = crud.get_transcript(db, meeting_id)
            messages = crud.get_messages(db, meeting_id)
            
            info = f"""
### 📋 {meeting.title}

**Status:** `{meeting.status}`  
**Tài liệu:** {len(docs)} files  
**Transcript:** {'✅' if transcript else '❌'} ({transcript.word_count if transcript else 0} từ)  
**Tin nhắn:** {len(messages)} messages  

{meeting.description if meeting.description else ''}
"""
            return f"✅ Đã load cuộc họp ID={meeting.id}", info
        finally:
            db.close()
    except Exception as e:
        return f"❌ Lỗi: {e}", ""

# =========================
# GRADIO UI - NotebookLM Style
# =========================

# Custom CSS
custom_css = """
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 0 20px;
}
.row, .block {
    width: 100%;
}
.tab-nav button {
    font-size: 16px;
    font-weight: 500;
}
.meeting-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.upload-zone {
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 30px;
    background: #f8f9ff;
}
.chat-container {
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(title="Meeting Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    with gr.Row(elem_classes=["meeting-header"]):
        gr.Markdown("""
        # 📝 Meeting Assistant
        ### Powered by AI - NotebookLM Style
        Tạo cuộc họp, upload tài liệu, ghi âm realtime, và hỏi đáp thông minh
        """)
    
    with gr.Tabs():
        # ==================== TAB 1: SETUP ====================
        with gr.Tab("📋 Thiết lập", id="setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🆕 Tạo cuộc họp mới")
                    meeting_title = gr.Textbox(
                        label="Tiêu đề cuộc họp",
                        placeholder="VD: Họp kế hoạch Q1 2025",
                        lines=1
                    )
                    meeting_desc = gr.Textbox(
                        label="Mô tả (tùy chọn)",
                        placeholder="Thảo luận kế hoạch kinh doanh và mục tiêu...",
                        lines=3
                    )
                    create_btn = gr.Button("➕ Tạo cuộc họp mới", variant="primary", size="lg")
                    create_status = gr.Markdown("")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Hoặc mở cuộc họp có sẵn")
                    refresh_btn = gr.Button("🔄 Tải danh sách", size="sm")
                    meeting_select = gr.Dropdown(
                        label="Chọn cuộc họp",
                        choices=[],
                        interactive=True
                    )
                    load_btn = gr.Button("📂 Mở cuộc họp", variant="secondary")
                    load_status = gr.Markdown("")
                    meeting_info_box = gr.Markdown("")
            
            # Upload section (hidden by default)
            with gr.Group(visible=False) as upload_section:
                gr.Markdown("### 📎 Upload tài liệu")
                with gr.Row(elem_classes=["upload-zone"]):
                    with gr.Column():
                        file_input = gr.File(
                            label="Chọn tài liệu",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt"]
                        )
                        upload_btn = gr.Button("📤 Upload & Phân tích", variant="primary", size="lg")
                
                upload_status = gr.Markdown("")
                
                with gr.Accordion("🔍 Xem Meeting Context", open=False):
                    context_box = gr.Textbox(
                        label="Context được tạo từ tài liệu",
                        lines=8,
                        interactive=False
                    )
        
        # ==================== TAB 2: RECORDING ====================
        with gr.Tab("🎙️ Ghi âm & Hỏi đáp", id="recording"):
            # Control buttons
            with gr.Row():
                with gr.Column(scale=1):
                    start_btn = gr.Button("▶️ Bắt đầu ghi âm", variant="primary", size="lg")
                with gr.Column(scale=1):
                    stop_btn = gr.Button("⏹️ Dừng ghi âm", variant="stop", size="lg")
                with gr.Column(scale=2):
                    status_box = gr.Markdown("_Chưa bắt đầu_")
            
            # Main content: 3 columns
            with gr.Row(equal_height=True):
                # Left: Transcript
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Transcript")
                    transcript_display = gr.Textbox(
                        show_label=False,
                        placeholder="Transcript sẽ hiển thị ở đây khi bắt đầu ghi âm...",
                        lines=35,
                        interactive=False,
                        max_lines=35
                    )
                
                # Center: Chat
                with gr.Column(scale=3, elem_classes=["chat-container"]):
                    gr.Markdown("### 💬 Hỏi đáp")
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=700,
                        bubble_full_width=False,
                        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")
                    )
                    with gr.Row():
                        chat_msg = gr.Textbox(
                            show_label=False,
                            placeholder="💭 Đặt câu hỏi về cuộc họp hoặc tài liệu...",
                            lines=2,
                            scale=9
                        )
                        send_btn = gr.Button("📤", scale=1, variant="primary")
                
                # Right: Summary
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Tóm tắt & Insights")
                    summary_display = gr.Textbox(
                        show_label=False,
                        placeholder="Các đoạn tóm tắt từ AI sẽ xuất hiện ở đây...",
                        lines=35,
                        interactive=False,
                        max_lines=35
                    )
    
    # ==================== EVENT HANDLERS ====================
    
    # Tab 1 events
    create_btn.click(
        fn=create_meeting,
        inputs=[meeting_title, meeting_desc],
        outputs=[create_status, upload_section]
    )
    
    refresh_btn.click(
        fn=load_meetings,
        outputs=[meeting_select]
    )
    
    load_btn.click(
        fn=select_meeting,
        inputs=[meeting_select],
        outputs=[load_status, meeting_info_box]
    )
    
    upload_btn.click(
        fn=upload_documents,
        inputs=[file_input],
        outputs=[upload_status, context_box]
    )
    
    # Tab 2 events
    start_btn.click(
        fn=start_recording,
        outputs=[transcript_display, summary_display, status_box]
    )
    
    stop_btn.click(
        fn=stop_recording,
        outputs=[status_box]
    )
    
    # Polling for real-time updates
    timer = gr.Timer(value=0.3, active=True)
    timer.tick(
        fn=poll_ui,
        outputs=[transcript_display, summary_display]
    )
    
    # Chat events
    send_btn.click(
        fn=chat_qa,
        inputs=[chatbot, chat_msg],
        outputs=[chatbot, chat_msg]
    )
    
    chat_msg.submit(
        fn=chat_qa,
        inputs=[chatbot, chat_msg],
        outputs=[chatbot, chat_msg]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
