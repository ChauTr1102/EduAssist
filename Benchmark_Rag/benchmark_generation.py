import json
import asyncio
import re
import string
from collections import Counter
from tqdm import tqdm
from api.services.vcdb_faiss import VectorStore
from api.services.local_llm import LanguageModelOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# --- CẤU HÌNH ---
# Đảm bảo đường dẫn đúng với máy bạn
BENCHMARK_PATH = "benchmark_questions.jsonl"
VECTOR_DB_PATH = "../vectorstores/Benchmark_rag"
MODEL_EMBEDDING = "Alibaba-NLP/gte-multilingual-base"
LLM_MODEL = "shmily_006/Qw3:4b_4bit"

# Test khoảng 50-100 câu thôi vì chạy LLM rất lâu
NUM_TEST_SAMPLES = 1000


# Hàm chuẩn hóa văn bản
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(của|là|những|các|một|cái|thì|mà)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Hàm tính F1
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# Hàm tính Exact Match
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


async def main():
    print("1. Loading Models...")
    # Dùng temperature thấp để model trả lời sát thực tế nhất
    llm = LanguageModelOllama(LLM_MODEL, temperature=0.1)

    model_embedding = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs={"trust_remote_code": True}
    )

    # Setup Vector Store thủ công để đảm bảo đúng cấu hình
    print("2. Setting up Retrievers (0.5/0.5)...")
    db = FAISS.load_local(VECTOR_DB_PATH, model_embedding, allow_dangerous_deserialization=True)
    cosine_retriever = db.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(list(db.docstore._dict.values()))
    bm25_retriever.k = 5

    # Cấu hình Hybrid chuẩn từ kết quả tuning của bạn
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, cosine_retriever],
        weights=[0.5, 0.5]
    )

    print(f"3. Loading {NUM_TEST_SAMPLES} questions...")
    questions = []
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= NUM_TEST_SAMPLES: break
            questions.append(json.loads(line))

    print("4. Starting Generation Benchmark...")
    f1_total = 0
    em_total = 0

    # Prompt ép trả lời ngắn gọn để dễ tính điểm
    # Lưu ý: Prompt này chỉ dùng để benchmark, không phải prompt chatbot thật
    BENCHMARK_PROMPT = """Dựa vào tài liệu sau, trả lời ngắn gọn câu hỏi đúng vào trọng tâm, không giải thích thêm.
Nếu không có thông tin, trả về "Không biết".

Tài liệu:
{context}

Câu hỏi: {question}
Trả lời:"""

    for item in tqdm(questions):
        question = item['question']
        ground_truth = item['ground_truth_answer']

        # 1. Retrieve
        docs = await ensemble_retriever.ainvoke(question)
        context_str = "\n".join([d.page_content for d in docs[:5]])

        # 2. Generate
        prompt = BENCHMARK_PROMPT.format(context=context_str, question=question)
        prediction = await llm.async_generate(prompt)

        # 3. Score
        f1 = f1_score(prediction, ground_truth)
        em = exact_match_score(prediction, ground_truth)

        f1_total += f1
        em_total += em

        # Debug: In thử 1 vài câu xem model trả lời thế nào
        # if f1 < 0.5:
        #    print(f"\n[Low Score] Q: {question}\nTrue: {ground_truth}\nPred: {prediction}\n")

    print("\n" + "=" * 30)
    print("🤖 GENERATION RESULTS")
    print("=" * 30)
    print(f"Samples: {NUM_TEST_SAMPLES}")
    print(f"✅ Avg F1-Score: {f1_total / NUM_TEST_SAMPLES:.2%}")
    print(f"✅ Exact Match:  {em_total / NUM_TEST_SAMPLES:.2%}")
    print("=" * 30)


if __name__ == "__main__":
    asyncio.run(main())