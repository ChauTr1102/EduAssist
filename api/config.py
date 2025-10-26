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

# Retriever
SEARCH_KWARGS = {'k': 25, 'score_threshold': 0.01, 'sorted': True}
SEARCH_TYPE = "similarity_score_threshold"

VECTOR_DATABASE = "./vectorstores/"

SYSTEM_DOCUMENT = "./data/data_system"
USER_DOCUMENT = "./data/data_user"

# Load data
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

MODEL_EMBEDDING = "AITeamVN/Vietnamese_Embedding"
