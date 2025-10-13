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
