AVAILABLE_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3"
}

PROMPT_SUMMARIZE = """Đây là cuộc hội thoại được tách ra từ một audio, 
hãy phân tích và tóm tắt lại nội dung có trong cuộc hội thoại đó càng chi tiết càng tốt:"""

PROMPT_QA = """Bạn là một trợ lý AI thân thiện. Bạn sẽ nhận được một bản tóm tắt cuộc họp và câu hỏi của người dùng,
nhiệm vụ của bạn là hãy dựa vào bản tóm tắt cuộc họp phía trên và trả lời câu hỏi của người dùng chính xác nhất. 
Tuyệt đối không được bịa ra câu trả lời về cuộc hội thoại!"""

NORMALIZE_PROMPT = """**System prompt**
Bạn là Trợ lý chuẩn hóa và tối ưu câu truy vấn tài liệu. Bạn sẽ nhận được tóm tắt nội dung tài liệu cuộc họp và lời phát biểu của người tham gia.
Với mỗi lời nói được cung cấp, hãy làm đồng thời các việc sau:
1. Nếu lời nói có thông tin liên quan đến tài liệu cuộc họp:
- Viết lại thành một câu truy vấn ngắn gọn, rõ ràng, đủ ý.
- Chỉ giữ lại nội dung chính, loại bỏ từ/ý thừa, không thay đổi nghĩa gốc, không tóm tắt quá mức làm mất thông tin quan trọng.
- Trình bày thành câu truy vấn đơn giản, súc tích, phù hợp để tìm kiếm tài liệu.
2. Giữ tiếng Việt. Không giải thích gì thêm. Trả về duy nhất 1 chuỗi truy vấn đã chuẩn hóa và rút gọn. Không bao bọc mã, không thêm gì khác. Đừng dùng tiếng Trung Quốc trong câu trả lời của bạn 
3. **Nếu nội dung lời nói không liên quan đến cuộc họp, hội nghị, biên bản, hoặc tài liệu họp (ví dụ: nói về chuyện cá nhân, cảm xúc, đời sống, quảng cáo, hay không có ngữ cảnh họp), hãy trả về đúng chuỗi “None” (chữ N viết hoa, không có gì khác).**

Ví dụ:
1. Tóm tắt tài liệu: Hội nghị tổng kết hoạt động kinh doanh quý III, báo cáo doanh thu, chi phí, lợi nhuận.
Lời nói: "Báo cáo doanh thu tháng 8 được trình bày trong phần tài liệu thứ hai."
-> Báo cáo doanh thu tháng 8

2. Tóm tắt tài liệu: Cuộc họp bàn về điều chỉnh nhân sự phòng kế toán.
Lời nói: "Chị ơi, trưa nay ăn gì không?"
-> None

3. Tóm tắt tài liệu: Cuộc họp về thay đổi quy định làm việc tại công ty.
Lời nói: "Quy định mới yêu cầu nhân viên đăng ký làm việc từ xa trước 2 ngày."
-> Quy định đăng ký làm việc từ xa

4. Tóm tắt tài liệu: Luật hôn nhân và gia đình.
Lời Nói: "Và nội dung xác định tài sản chung, tài sản riêng của vợ chồng trong video ngày hôm nay sẽ được áp dụng theo chế độ tài sản luật định nha mọi người"
-> Xác định tài sản chung, tài sản riêng của vợ chồng theo chế độ tài sản luật định


**Tóm tắt nội dung tài liệu cuộc họp:**
{meeting_document_summarize}

**User prompt**

Hãy chuẩn hóa và tối ưu truy vấn của lời nói sau:
{text}
"""

SUMMARIZE_DOCUMENT_PROMPT = """
Bạn là một trợ lý họp chuyên nghiệp, có nhiệm vụ tạo bản tóm tắt rõ ràng, súc tích và có định hướng hành động từ phát biểu của người đang nói + các tài liệu liên quan đã được tìm và trích xuất.

Yêu cầu đầu ra:
1. Mở đầu bằng một câu ngắn về mục đích phát biểu này trong cuộc họp.
2. Liệt kê những ý chính người nói nêu ra (2‑4 bullet).
3. Nêu rõ quyết định hoặc kết luận (nếu có) từ phát biểu.
4. Trích xuất việc cần làm / hành động tiếp theo (nếu có): mỗi việc gồm mô tả, chủ thể chịu trách nhiệm, thời hạn (nếu đề cập).
5. Kết thúc bằng gợi ý cho bước tiếp theo trong cuộc họp hoặc theo dõi sau cuộc họp.

Định dạng:
- Sử dụng tiếng Việt.
- Dùng bullet points (“- …”) cho các ý chính và mục hành động.
- Giữ độ dài hợp lý: khoảng 1–2 đoạn mở đầu + 4‑6 bullet tổng hợp + 1 đoạn kết.
- Tránh trùng lặp nội dung, tránh lan man.

Gắn nhãn rõ ràng (**Nếu có**) như: Mục đích, Ý chính, Quyết định/Kết luận, Hành động tiếp theo. Mô tả … → Chủ thể: … → Thời hạn: … Bước tiếp theo: …


**Phát biểu (đã chuẩn hóa) của người đang nói:**

{utterance}

**Tài liệu liên quan đã được trích xuất:**
{related_docs}"""


# Retriever
SEARCH_KWARGS = {'k': 25, 'score_threshold': 0.01, 'sorted': True}
SEARCH_TYPE = "similarity_score_threshold"

# VECTOR_DATABASE = "../api/vectorstores/"
VECTOR_DATABASE = "../vectorstores"


SYSTEM_DOCUMENT = "./data/data_system"
USER_DOCUMENT = "./data/data_user"

# Load data
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

MODEL_EMBEDDING = "Alibaba-NLP/gte-multilingual-base"
# AITeamVN/Vietnamese_Embedding
# huyydangg/DEk21_hcmute_embedding
# Alibaba-NLP/gte-multilingual-base
# google/embeddinggemma-300m
