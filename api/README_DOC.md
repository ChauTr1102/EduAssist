**Cấu trúc thư mục `api/` và mô tả nhanh các file**

Mô tả này chỉ giải thích vai trò / mục đích của từng file hoặc thư mục — KHÔNG giải thích chi tiết bên trong mã.

**Thư mục gốc `api/`**
- `__init__.py`: Tệp khởi tạo gói Python cho `api`.
- `asr_punct_norm_summarize-pipeline.py`: Script/entrypoint liên quan tới pipeline xử lý ASR, chấm câu, chuẩn hoá và tóm tắt (kịch bản tích hợp các bước xử lý).
- `config.py`: Tập trung các cấu hình chung cho module `api` (đường dẫn, hằng số, tham số cấu hình).
- `indexing_doc.py`: Công cụ/tiện ích phục vụ việc lập chỉ mục tài liệu (indexing) cho hệ thống tìm kiếm hoặc vector store.
- `private_config.py`: Chứa các cấu hình riêng tư/nhạy cảm (ví dụ khóa, thông tin môi trường) — thường không đẩy lên public.
- `test_embed_utterance_realtime.py`: Tập tin thử nghiệm liên quan tới embedding hoặc nhúng câu nói trong môi trường realtime.
- `routes/`: Thư mục chứa các định nghĩa route/HTTP server cho API.
- `services/`: Thư mục chứa các service hỗ trợ (ASR, LLM, chuyển đổi audio, xử lý chấm câu, v.v.).
- `utils/`: Thư mục tiện ích hỗ trợ (hàm chung, xử lý dữ liệu, v.v.).

**Thư mục `api/routes/`**
- `__init__.py`: Khởi tạo gói con `routes`.
- `routes.py`: Định nghĩa các route (endpoints) của API — nơi ánh xạ các đường dẫn HTTP tới handlers.
- `server.py`: Cấu hình và khởi chạy server (ví dụ khởi tạo app, middleware, lắng nghe cổng).

**Thư mục `api/services/` (mô tả các file chính)**
- `__init__.py`: Khởi tạo gói con `services`.
- `chunkformer_stt.py`: Service tích hợp với mô hình Chunkformer để thực hiện STT (speech-to-text).
- `local_llm.py`: Phiên bản LLM chạy cục bộ hoặc adapter cho mô hình local.
- `lockcahead.txt`: Tệp văn bản/cấu hình nội bộ; có thể liên quan tới khóa/ghi chú triển khai (dạng tài liệu trợ giúp).
- `punctuation_processing.py`: Tiện ích/chức năng phục vụ xử lý chấm câu trên văn bản sau ASR.
- `requirements.txt`: Danh sách phụ thuộc Python cần cho phần `services` (dành cho môi trường hoặc cài đặt riêng).
- `vcdb_faiss.py`: Service/tiện ích liên quan tới FAISS vector store (tạo, truy vấn, quản lý index).
- `video_to_audio_convert.py`: Công cụ chuyển đổi video sang audio (tách âm thanh để đưa vào pipeline ASR).

**Thư mục `api/utils/` (tổng quan)**
- Gồm các tiện ích giúp xử lý câu, định dạng thời gian, quản lý bộ nhớ GPU, và các hàm dùng chung khác (ví dụ `raw_utterances_processing.py`, `time_format.py`, `utils_gpu_mem.py`).


**Hàm / Entrypoint chính (dùng cho tài liệu vận hành)**
- `api/asr_punct_norm_summarize-pipeline.py` (đây là luồng cũ, ko cần xem nếu ko thích, luồng chính được áp dụng trong file fronend/new_ui.py) :
  - `start_async_loop()`: khởi động vòng lặp asyncio nền.
  - `stop_async_loop()`: dừng vòng lặp asyncio nền.
  - `run_async(coro, timeout)`: thực thi coroutine bất đồng bộ với timeout tùy chọn.
  - `start_workers(num_workers)`, `worker_loop(worker_id)`: tạo và quản lý worker cho pipeline xử lý.
  - `start_summarizer()`, `summarizer_loop()`: entrypoint cho luồng tóm tắt văn bản.
  - `build_summary_prompt(utterance, docs)`: xây dựng prompt tóm tắt (entrypoint cho việc tạo prompt).

- `api/test_embed_utterance_realtime.py` (file test embedding transcript realtime):
  - `on_update(event, payload, full)`: callback/entrypoint nhận sự kiện realtime.
  - `just_print(event, payload, full)`: hàm trợ giúp để in payload trong thử nghiệm.

- `api/routes/routes.py`(phần route của fastapi chưa cập nhật, ko cần xem file này và server.py) :
  - `home()`: endpoint gốc (healthcheck / trang chỉ dẫn).
  - `speech_to_text(audio_path)`: endpoint nhận audio và trả kết quả STT.
  - `chat_with_ai(request)`, `summarize_gemini(script)`, `chat_gemini(user_input)`: các endpoint tương tác với LLM/AI.
  - `extract_audio_from_path(request)`: endpoint tách audio từ video theo đường dẫn.

- `api/routes/server.py`:
  - `_loop_worker(loop)`, `run_async(coro, timeout)`: tiện ích chạy coroutine trong vòng lặp nền.
  - `worker_loop(worker_id)`, `summarizer_loop()`: luồng worker tương tự pipeline.
  - Các phương thức quản lý WebSocket / session: `register(session_id)`, `unregister(session_id)`, `publish_to(session_id, payload)`, `_safe_send_json(data)`, `close()`.

- `api/services/vcdb_faiss.py` (class chính và method):
  - `hybrid_search(question)`: tìm kiếm kết hợp (semantic + chính xác).
  - `semantic_chunking(file_path)`: thực hiện chunking ngữ nghĩa/phiên bản bất đồng bộ.
  - `create_vectorstore(chunks)`, `faiss_save_local(db, type_id)`, `merge_to_vectorstore(old_db, new_db, meeting_id, type_id)`, `delete_from_vectorstore(file_name, meeting_id, type_id)`: các entrypoint quản lý vectorstore.
  - `add_transcript(transcript, start, end)`, `add_cache(transcript)`: thêm transcript hoặc cache vào vectorstore.
  - `is_already_retrieved(text, top_k, similarity_threshold)`: kiểm tra đã được truy xuất trước đó hay chưa.

- `api/services/video_to_audio_convert.py`:
  - `extract_audio(video_path, output_dir)`: entrypoint chuyển đổi video -> audio và trả đường dẫn audio kết quả.

- `api/services/local_llm.py`:
  - `__init__(...)`: khởi tạo adapter LLM cục bộ.
  - `_endpoint()`, `_get_client()`, `aclose()`: quản lý client/endpoint.
  - `prompt_process(sentence)`, `normalize_text(meeting_document_summarize, transcript)`: entrypoint tiền xử lý prompt và chuẩn hoá văn bản cho LLM.

- `api/services/Vietnamese-Inverse-Text-Normalization/inverse_normalize.py`:
  - `inverse_normalize(s, verbose=False)`: entrypoint thực hiện ngược chuẩn hoá văn bản tiếng Việt.

- `api/utils/*` (một số hàm tiện ích quan trọng):
  - `raw_utterances_processing.py`: `parse_transcript_to_utterances`, `utterances_to_documents`, `mmssms_to_seconds` — chuyển đổi transcript thô sang utterances / documents.
  - `time_format.py`: `ms_to_hms_pad(ms)` — chuyển đổi thời gian.
  - `utils_gpu_mem.py`: `get_gpu_memory_mb(device_index)`, `detect_device_index()` — kiểm tra tài nguyên GPU.

Lưu ý: Danh sách trên tập trung vào các hàm/entrypoint quan trọng cho vận hành; không mô tả chi tiết triển khai bên trong hàm.

**Mô tả thư mục `vectorstores/`**
- Vị trí: `vectorstores/` (thư mục gốc của repository). Đây là nơi lưu trữ các chỉ mục vector (ví dụ FAISS) và dữ liệu liên quan để tra cứu/ truy vấn ngữ nghĩa.
- Cấu trúc chung:
  - `vectorstores/<collection>/documents/` để lưu embedding của tài liệu cuộc họp 
  - `vectorstores/<collection>/transcripts/`: để lưu embedding của transcript cuộc họp.
  - `vectorstores/<collection>/cache/`: để lưu embedding của hệ thống duplicated prompt detection (prompt cache).
- Mục đích vận hành:
  - Lưu trữ chỉ mục để nâng cao tốc độ truy vấn semantic retrieval.
  - Các index này có thể được tải lại khi khởi động service để tránh re-indexing tốn thời gian.
  - Khi cập nhật dữ liệu (thêm/merge/xoá), cần gọi các hàm quản lý vectorstore (ví dụ từ `vcdb_faiss.py`) để cập nhật index tương ứng.
- Lưu ý vận hành:
  - Không xoá file `index.faiss` trực tiếp nếu không muốn mất dữ liệu tìm kiếm; hãy sử dụng chức năng `delete_from_vectorstore` / `merge_to_vectorstore` để quản lý an toàn.
  - Khi chuyển môi trường (ví dụ dev → prod), đồng bộ toàn bộ thư mục collection để giữ tính toàn vẹn index.

**Mô tả nhanh `frontend/new_ui.py`** (đây là luồng chính gộp vào 1 file, bao gồm cả UI đơn giản)
- Mục đích: file UI/khung điều phối frontend có nhiều entrypoint dùng để chạy worker, xử lý ASR/embedding/summarizer và giao tiếp với backend.
- Entrypoint chính (dành cho tài liệu vận hành):
  - `init_itn_model(itn_model_dir)`: khởi tạo mô-đun ITN (inverse text normalization) nếu cần.
  - `start_async_loop()`, `stop_async_loop()`, `run_async(coro, timeout)`: quản lý vòng lặp asyncio nền cho UI.
  - `start_workers(num_workers)`, `worker_loop(worker_id)`: khởi chạy worker xử lý nền.
  - `embedding_worker()`, `start_embedding_worker()`: worker tạo embedding và enqueue sang vectorstore.
  - `start_summarizer()`, `summarizer_loop()`: luồng tóm tắt văn bản.
  - `enqueue_rag_with_overlap(new_commit)`: enqueue tài liệu/commit mới để xử lý RAG (retrieval-augmented generation) với overlap.
  - `on_update(event, payload)`: callback nhận các sự kiện UI/realtime.
  - `asr_worker()`, `start_asr()`, `stop_asr()`: quản lý worker và vòng đời ASR.
  - `poll_ui()`: vòng lặp polling cập nhật UI.
  - `chat_qa(history, message)`: entrypoint gửi câu hỏi chat/QA và nhận phản hồi từ backend/LLM.
- Gợi ý vận hành:
  - Các hàm `start_...` và `stop_...` là điểm bắt đầu/ kết thúc cho các dịch vụ nền; khi deploy hoặc restart UI, gọi đúng sequence để đảm bảo tài nguyên (GPU, client) được khởi tạo và giải phóng.
  - Nếu cần debug luồng nền, bật logging xung quanh `worker_loop`, `embedding_worker` và `asr_worker` để theo dõi tiến trình xử lý.
