from api.services.vcdb_faiss import VectorStore
from api.config import *

faiss = VectorStore("LichSuDangVietNam")

chunks = faiss.recursive_chunking("/home/bojjoo/Downloads/lich_su_dang_cong_san_Vietnam.pdf")

db = faiss.create_vectorstore(chunks)

faiss.faiss_save_local(db)