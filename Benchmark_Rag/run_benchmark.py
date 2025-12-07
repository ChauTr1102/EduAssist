import json
import time
import asyncio
import numpy as np
from tqdm import tqdm
from api.services.vcdb_faiss import VectorStore  # Sửa path import cho đúng với project của bạn
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# CẤU HÌNH
KB_PATH = "knowledge_base.jsonl"
BENCHMARK_PATH = "benchmark_questions.jsonl"
MODEL_EMBEDDING = "Alibaba-NLP/gte-multilingual-base"
VECTOR_DB_PATH = "../vectorstores/Benchmark_rag"

NUM_QUESTIONS_TO_TEST = 1000  # Test 200 câu cho mỗi cấu hình


async def main():
    print("⏳ 1. Loading Models & Vector DB...")
    model_embedding = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs={"trust_remote_code": True}
    )

    # Init VectorStore (Lưu ý: class VectorStore của bạn phải tự load DB bên trong __init__ hoặc gán thủ công như bạn đã làm ở các phiên bản trước)
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

        recall_at_1 = 0  # <--- Thêm biến đếm Recall@1
        recall_at_5 = 0
        mrr_score = 0

        # Dùng tqdm để hiện thanh tiến trình cho mỗi config
        for item in tqdm(questions, desc=f"Eval {w_bm25}/{w_cos}", leave=False):
            query = item['question']
            ground_truth_id = item['ground_truth_doc_id']

            # Gọi hàm search với trọng số động
            retrieved_docs = await vector_store.search_for_benchmark(
                query, k=5,
                weight_bm25=w_bm25,
                weight_cosine=w_cos
            )

            retrieved_ids = [doc.metadata.get('doc_id') for doc in retrieved_docs]

            # 1. Tính Recall@5 (Có trong top 5)
            if ground_truth_id in retrieved_ids:
                recall_at_5 += 1

                # Tính MRR
                rank = retrieved_ids.index(ground_truth_id) + 1
                mrr_score += 1 / rank

            # 2. Tính Recall@1 (Có ngay ở vị trí đầu tiên)
            if retrieved_ids and ground_truth_id == retrieved_ids[0]:  # <--- Logic tính Recall@1
                recall_at_1 += 1

        # Tổng kết cho config này
        score_recall_1 = recall_at_1 / len(questions)  # <---
        score_recall_5 = recall_at_5 / len(questions)
        score_mrr = mrr_score / len(questions)

        final_results.append({
            "bm25": w_bm25,
            "cosine": w_cos,
            "recall@1": score_recall_1,  # <--- Lưu vào dict
            "recall@5": score_recall_5,
            "mrr": score_mrr
        })

        print(f"   -> R@1: {score_recall_1:.2%} | R@5: {score_recall_5:.2%} | MRR: {score_mrr:.4f}")

    # --- IN BẢNG KẾT QUẢ CUỐI CÙNG ---
    print("\n" + "=" * 70)  # Kéo dài bảng ra chút cho đẹp
    # Thêm cột Recall@1 vào Header
    print(f"{'BM25':<10} | {'Cosine':<10} | {'Recall@1':<10} | {'Recall@5':<10} | {'MRR':<10}")
    print("-" * 70)

    # Tìm best result (Vẫn dựa trên Recall@5 hoặc MRR để chọn best)
    best_score = -1
    best_config = None

    for res in final_results:
        # In thêm cột Recall@1
        print(
            f"{res['bm25']:<10} | {res['cosine']:<10} | {res['recall@1']:.2%}     | {res['recall@5']:.2%}     | {res['mrr']:.4f}")

        # Chọn best config dựa trên Recall@5 (hoặc bạn có thể đổi thành MRR tùy ý muốn)
        if res['recall@5'] > best_score:
            best_score = res['recall@5']
            best_config = res

    print("-" * 70)
    print(f"🏆 BEST CONFIGURATION: BM25={best_config['bm25']} / Cosine={best_config['cosine']}")
    print(f"   With Recall@1: {best_config['recall@1']:.2%}")  # In ra kết quả best
    print(f"   With Recall@5: {best_config['recall@5']:.2%}")
    print(f"   With MRR:      {best_config['mrr']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())