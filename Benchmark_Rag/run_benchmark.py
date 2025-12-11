import json
import time
import asyncio
import numpy as np
from tqdm import tqdm
from api.services.vcdb_faiss import VectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# CẤU HÌNH
KB_PATH = "knowledge_base.jsonl"
BENCHMARK_PATH = "benchmark_questions.jsonl"
MODEL_EMBEDDING = "Alibaba-NLP/gte-multilingual-base"
VECTOR_DB_PATH = "../vectorstores/Benchmark_rag"

# Số lượng câu hỏi muốn test (Lưu ý: 19000 câu sẽ chạy rất lâu, nếu test nhanh nên giảm xuống)
NUM_QUESTIONS_TO_TEST = 19000


async def main():
    print("⏳ 1. Loading Models & Vector DB...")
    model_embedding = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs={"trust_remote_code": True}
    )

    # Init VectorStore
    vector_store = VectorStore("Benchmark_rag", model_embedding)

    # Load câu hỏi
    questions = []
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= NUM_QUESTIONS_TO_TEST: break
            questions.append(json.loads(line))

    # --- BẮT ĐẦU VÒNG LẶP TUNING ---
    weights_to_test = []
    for i in range(1, 10):
        w_bm25 = round(i * 0.1, 1)
        w_cos = round(1.0 - w_bm25, 1)
        weights_to_test.append((w_bm25, w_cos))

    final_results = []

    print(f"\n🚀 Starting Hyperparameter Tuning on {len(questions)} questions...")

    for w_bm25, w_cos in weights_to_test:
        print(f"\n⚙️ Testing Config: BM25={w_bm25} | Cosine={w_cos}")

        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_8 = 0  # <--- Mới
        recall_at_10 = 0  # <--- Mới
        mrr_score = 0

        for item in tqdm(questions, desc=f"Eval {w_bm25}/{w_cos}", leave=False):
            query = item['question']
            ground_truth_id = item['ground_truth_doc_id']

            # QUAN TRỌNG: Phải lấy k=10 thì mới tính được Recall@10
            retrieved_docs = await vector_store.search_for_benchmark(
                query, k=10,
                weight_bm25=w_bm25,
                weight_cosine=w_cos
            )

            retrieved_ids = [doc.metadata.get('doc_id') for doc in retrieved_docs]

            # Kiểm tra xem đáp án đúng có trong list tìm thấy không
            if ground_truth_id in retrieved_ids:
                # Lấy vị trí (index) của đáp án đúng (bắt đầu từ 0)
                rank_index = retrieved_ids.index(ground_truth_id)

                # Recall@1: Nằm ở vị trí đầu tiên (index 0)
                if rank_index == 0:
                    recall_at_1 += 1

                # Recall@5: Nằm trong 5 vị trí đầu (index 0-4)
                if rank_index < 5:
                    recall_at_5 += 1

                # Recall@8: Nằm trong 8 vị trí đầu (index 0-7)
                if rank_index < 8:
                    recall_at_8 += 1

                # Recall@10: Nằm trong 10 vị trí đầu (index 0-9)
                # Vì ta đã check `if ground_truth_id in retrieved_ids` và k=10 nên chắc chắn đúng
                if rank_index < 10:
                    recall_at_10 += 1

                # Tính MRR
                mrr_score += 1 / (rank_index + 1)

        # Tổng kết cho config này
        scores = {
            "bm25": w_bm25,
            "cosine": w_cos,
            "recall@1": recall_at_1 / len(questions),
            "recall@5": recall_at_5 / len(questions),
            "recall@8": recall_at_8 / len(questions),  # <---
            "recall@10": recall_at_10 / len(questions),  # <---
            "mrr": mrr_score / len(questions)
        }

        final_results.append(scores)

        print(
            f"   -> R@1: {scores['recall@1']:.2%} | R@5: {scores['recall@5']:.2%} | R@10: {scores['recall@10']:.2%} | MRR: {scores['mrr']:.4f}")

    # --- IN BẢNG KẾT QUẢ CUỐI CÙNG ---
    # Kéo dài bảng ra để chứa đủ cột
    print("\n" + "=" * 100)
    header = f"{'BM25':<6} | {'Cosine':<6} | {'R@1':<8} | {'R@5':<8} | {'R@8':<8} | {'R@10':<8} | {'MRR':<8}"
    print(header)
    print("-" * 100)

    # Tìm best result (Vẫn dựa trên Recall@5 làm tiêu chuẩn, hoặc bạn đổi sang R@10 tùy ý)
    best_score = -1
    best_config = None

    for res in final_results:
        row = f"{res['bm25']:<6} | {res['cosine']:<6} | {res['recall@1']:.2%}   | {res['recall@5']:.2%}   | {res['recall@8']:.2%}   | {res['recall@10']:.2%}   | {res['mrr']:.4f}"
        print(row)

        if res['recall@5'] > best_score:
            best_score = res['recall@5']
            best_config = res

    print("-" * 100)
    print(f"🏆 BEST CONFIGURATION (Based on R@5): BM25={best_config['bm25']} / Cosine={best_config['cosine']}")
    print(f"   Recall@1:  {best_config['recall@1']:.2%}")
    print(f"   Recall@5:  {best_config['recall@5']:.2%}")
    print(f"   Recall@8:  {best_config['recall@8']:.2%}")
    print(f"   Recall@10: {best_config['recall@10']:.2%}")
    print(f"   MRR:       {best_config['mrr']:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())