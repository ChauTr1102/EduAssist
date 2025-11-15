from api.services.vcdb_faiss import VectorStore
from api.config import *

faiss = VectorStore("luat_hon_nhan_gia_dinh")

chunks = faiss.recursive_chunking("/home/bojjoo/Downloads/luat_hon_nhan_gia_dinh.pdf")

db = faiss.create_vectorstore(chunks)

faiss.faiss_save_local(db)