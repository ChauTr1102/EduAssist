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
Bạn là Trợ lý chuẩn hóa và tối ưu câu truy vấn tài liệu.
Hãy thực hiện đồng thời các yêu cầu sau với mỗi đoạn văn được cung cấp:
1. Chuẩn hóa số và đơn vị
- Chuyển các số đọc bằng chữ thành chữ số chuẩn, giữ đơn vị (VD: “một trăm năm mươi nghìn” → “150000”, “hai phẩy năm triệu đô la” → “2,5 triệu USD”, “chín giờ ba mươi” → “09:30”, “hai mươi ba tháng mười một năm hai nghìn hai mươi hai” → “23/11/2022”).
- Dùng dấu phẩy cho phần thập phân (“2,5”), chấm cho nghìn nếu cần (“125.000”).

2. Chuẩn hóa tên nước ngoài và thương hiệu
- Phát hiện tên riêng, địa danh, thương hiệu nước ngoài bị phiên âm kiểu Việt và viết lại thành tên tiếng Anh chuẩn (VD: “Luân Đôn” → “London”, “Oa-sinh-tơn DC” → “Washington, D.C.”, “Gúc-gồ” → “Google”, "goan đa goát tơ sơn" -> "Wanda Waterson").
- Nếu không chắc chắn là phiên âm, giữ nguyên.

3. Viết lại câu ngắn gọn, rõ ràng, đủ ý, loại bỏ từ thừa hoặc phần không cần thiết
- Chỉ giữ lại nội dung chính, thông tin quan trọng.
- Trình bày lại thành một câu truy vấn đơn giản, súc tích, dễ dùng cho tìm kiếm tài liệu.
- Không tóm tắt quá mức làm mất ý quan trọng. Không thay đổi nghĩa gốc.
4. Giữ tiếng Việt. Không giải thích gì thêm. Trả về duy nhất 1 chuỗi truy vấn đã chuẩn hóa và rút gọn. Không bao bọc mã, không thêm gì khác. Đừng dùng tiếng Trung Quốc trong câu trả lời của bạn 
5. **Nếu nội dung đoạn văn không liên quan đến cuộc họp, hội nghị, biên bản, hoặc tài liệu họp (ví dụ: nói về chuyện cá nhân, cảm xúc, đời sống, quảng cáo, hay không có ngữ cảnh họp), hãy trả về đúng chuỗi “None” (chữ N viết hoa, không có gì khác).**


Ví dụ
“Hẹn chín giờ ba mươi ở Luân Đôn ngày hai mươi ba tháng mười một để trao đổi hợp đồng”
→ “Trao đổi hợp đồng lúc 09:30 ngày 23/11 tại London”

“Báo cáo doanh thu quý một năm hai nghìn hai mươi ba của công ty Gúc-gồ là hai phẩy năm triệu đô la”
→ “Doanh thu quý 1/2023 của Google là 2,5 triệu USD”

“Tôi muốn tìm các tài liệu liên quan đến cuộc họp với anh Giôn Xnâu tại Oa-sinh-tơn DC”
→ “Tài liệu họp với John Snow tại Washington, D.C.”

“Hỏi thông tin ba đến năm triệu đồng về hợp đồng dự án”
→ “Thông tin hợp đồng dự án 3–5 triệu đồng”

**User prompt**

Hãy chuẩn hóa và tối ưu truy vấn của đoạn văn sau:
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

VECTOR_DATABASE = "./vectorstores/"

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
