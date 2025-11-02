from api.services.vcdb_faiss import VectorStore
from api.config import *

faiss = VectorStore("Baocaouyvienbochinhtri")

chunks = faiss.recursive_chunking("/home/bojjoo/Downloads/baocaouyvienbochinhtri.docx")

db = faiss.create_vectorstore(chunks)

faiss.faiss_save_local(db)