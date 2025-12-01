from api.services.vcdb_faiss import VectorStore
from api.config import *
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_embedding = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING, model_kwargs={"trust_remote_code": True})
faiss = VectorStore("luat_hon_nhan_gia_dinh", model_embedding)

chunks = faiss.recursive_chunking("/home/bojjoo/Downloads/luat_hon_nhan_gia_dinh.pdf")

db = faiss.create_vectorstore(chunks)

faiss.faiss_save_local(db, "documents")

print("Indexed documents")