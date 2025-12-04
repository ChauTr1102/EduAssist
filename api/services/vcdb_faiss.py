from api.services import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import FAISS
import os
from fastapi import UploadFile, File, Form
import shutil
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
import tiktoken
from api.utils.raw_utterances_processing import *

class VectorStore:
    def __init__(self, meeting_id: str, model_embedding):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                            length_function=len)
        # self.model_embedding = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING, google_api_key=openai_embedding_key)
        self.meeting_id = meeting_id
        self.model_embedding = model_embedding
        try:
            self.db = FAISS.load_local(f'{VECTOR_DATABASE}/{meeting_id}', self.model_embedding,
                                       allow_dangerous_deserialization=True)
            self.cosine_retriever = self.db.as_retriever(search_kwargs=SEARCH_KWARGS, search_type=SEARCH_TYPE)
            documents = list(self.db.docstore._dict.values())
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 25
        except Exception as e:
            print(f"Can't Load VectorDB: {e}")
            self.db = None
            self.cosine_retriever = None
            self.bm25_retriever = None

    async def hybrid_search(self, question):
        ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.cosine_retriever],
                                            weights=[0.5, 0.5])

        compressed_docs = await ensemble_retriever.ainvoke(question)
        results = []
        for doc in compressed_docs[:4]:
            start = doc.metadata.get('start_seconds', None)
            end = doc.metadata.get('end_seconds', None)
            results.append(f"[{start} - {end}] {doc.page_content}")
        content_text = "\n---\n".join(results)
        return content_text

    def recursive_chunking(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def recursive_chunking_text_speakers(self, raw_transcript, start, end):
        """Xử lý đoạn transcript chuẩn như sau:
00:00.858 --> 00:07.511
[SPEAKER_00]: {transcript}

00:07.531 --> 00:08.913
[SPEAKER_00]: {transcript}

00:08.934 --> 00:21.718
[SPEAKER_01]: {transcript}

00:44.952 --> 00:50.724
[SPEAKER_00]: {transcript}

        thành kiểu Document của Langchain:
[Document(metadata={'speaker': 'SPEAKER_00', 'start_seconds': 0.858, 'end_seconds': 7.511, 'duration_seconds': 6.653, 'turn_id': 1, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'SPEAKER_00', 'start_seconds': 7.531, 'end_seconds': 8.913, 'duration_seconds': 1.382, 'turn_id': 2, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'SPEAKER_01', 'start_seconds': 8.934, 'end_seconds': 21.718, 'duration_seconds': 12.784, 'turn_id': 3, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'SPEAKER_00', 'start_seconds': 44.952, 'end_seconds': 50.724, 'duration_seconds': 5.772, 'turn_id': 4, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}')]
 """


        utterances = parse_transcript_to_utterances(raw_transcript)
        docs = utterances_to_documents(utterances, conversation_id="hop_nextstart")

        # Cần review lại có cần không
        chunks = self.text_splitter.split_documents(docs)
        return chunks

    def recursive_chunking_text_no_speaker(self,transcript, start, end, idx):
        """Xử lý đoạn dictionary chuẩn như sau:
{"start": 0.858, "end":7.511, "text":{transcript}}, ...

        thành kiểu Document của Langchain:
[Document(metadata={'speaker': 'UNKNOWN', 'start_seconds': 0.858, 'end_seconds': 7.511, 'duration_seconds': 6.653, 'turn_id': 1, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'UNKNOWN', 'start_seconds': 7.531, 'end_seconds': 8.913, 'duration_seconds': 1.382, 'turn_id': 2, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'UNKNOWN', 'start_seconds': 8.934, 'end_seconds': 21.718, 'duration_seconds': 12.784, 'turn_id': 3, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}'),
 Document(metadata={'speaker': 'UNKNOWN', 'start_seconds': 44.952, 'end_seconds': 50.724, 'duration_seconds': 5.772, 'turn_id': 4, 'conversation_id': 'hop_nextstart'}, page_content='{transcript}')]
 """

        doc = utterances_to_documents_no_speakers(transcript, start, end, idx)

        # Cần review lại có cần không
        chunks = self.text_splitter.split_documents([doc])
        return chunks

    async def semantic_chunking(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        documents = loader.load()

        text_splitter = SemanticChunker(self.model_embedding, breakpoint_threshold_type='percentile',
                                        breakpoint_threshold_amount=95)
        chunks = await text_splitter.create_documents(texts=[document.page_content for document in documents],
                                                metadatas=[document.metadata for document in documents])
        return chunks

    # Lưu vào vectorstore
    def create_vectorstore(self, chunks):
        db = FAISS.from_documents(chunks, self.model_embedding)
        return db

    def faiss_save_local(self, db, type_id):
        db.save_local(f'{VECTOR_DATABASE}/{self.meeting_id}/{type_id}')

    # thêm vào vectorstore
    def merge_to_vectorstore(self, old_db, new_db, meeting_id):
        old_db.merge_from(new_db)
        old_db.save_local(f'{VECTOR_DATABASE}/{meeting_id}')
        return old_db

    # xóa khỏi vectorstore theo id của chunks
    def delete_from_vectorstore(self, file_name, meeting_id, type_id):
        # db_user, retriever_user = self.check_user_db(user_id)
        db_user = self.db
        docstore = db_user.docstore._dict
        key_delete = []
        for key, values in docstore.items():
            if values.metadata['source'].endswith(f"{file_name}"):
                key_delete.append(key)
        db_user.delete(key_delete)
        db_user.save_local(f"{VECTOR_DATABASE}/{meeting_id}/{type_id}")

    def add_transcript(self, transcript, start, end):
        idx = len(self.db.docstore._dict) if self.db is not None else 0
        chunks = self.recursive_chunking_text_no_speaker(transcript, start, end, idx)
        # bỏ vào vectorstore mới
        new_db = self.create_vectorstore(chunks)
        if self.db is not None: # Nếu đã có db, hợp nhất db cũ với db mới
            self.db = self.merge_to_vectorstore(self.db, new_db, self.meeting_id)

            # Cập nhật retriever
            self.cosine_retriever = self.db.as_retriever(search_kwargs=SEARCH_KWARGS, search_type=SEARCH_TYPE)
            documents = list(self.db.docstore._dict.values())
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 25
            return None
        else:  # Nếu chưa có db
            new_db.save_local(f'{VECTOR_DATABASE}/{self.meeting_id}')
            self.db = new_db
            # Cập nhật retriever
            self.cosine_retriever = self.db.as_retriever(search_kwargs=SEARCH_KWARGS, search_type=SEARCH_TYPE)
            documents = list(self.db.docstore._dict.values())
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 25
            return None

    def add_cache(self, transcript):
        chunk = cache_documenting(transcript)
        # bỏ vào vectorstore mới
        new_db = self.create_vectorstore([chunk])
        if self.db is not None: # Nếu đã có db, hợp nhất db cũ với db mới
            self.db = self.merge_to_vectorstore(self.db, new_db, self.meeting_id)
            return None
        else:  # Nếu chưa có db
            new_db.save_local(f'{VECTOR_DATABASE}/{self.meeting_id}')
            self.db = new_db
            return None

    # --- 2. Hàm kiểm tra novelty / trùng lặp ---
    def is_already_retrieved(self, text: str, top_k: int = 1, similarity_threshold: float = None) -> bool:
        """
        Kiểm tra xem text (đã được normalize / clean) đã từng được embed + lưu trong cache_faiss hay chưa.
        Dùng threshold phù hợp:
          - Cosine similarity: score >= similarity_threshold → coi là trùng.
        """
        text = text.strip()
        if not text:
            return False

        # Search nearest in cache
        if self.db is None:
            self.add_cache(text)
            return False

        try:
            results = self.db.similarity_search_with_relevance_scores(text, k=top_k)
        except Exception as e:
            print(f"[cache check] error in similarity_search_with_score: {e}")
            return False

        if not results:
            return False

        # results: list of (Document, score_or_distance)
        doc, score = results[0]

        if score >= similarity_threshold:
            return True

        # Nếu không xác định threshold → luôn coi là “chưa xử lý”
        return False

    # # upload file và lưu vào vectorstore faiss, lưu file vào folder của conversation_id
    # def upload_file(self, file: UploadFile = File(...), user_id: str = Form(...), folder_id: str = Form(...),
    #                 semantic_chunking: bool = Form(...)):
    #     name = file.filename
    #     type = "text_file"
    #     if name.endswith('.pdf') or name.endswith('docx'):
    #         # Lấy ra file size
    #         file.file.seek(0, os.SEEK_END)
    #         file_size = round(file.file.tell() / (1024 * 1024), 2)
    #         file.file.seek(0)
    #         result = sql_conn.save_file_detail(file.filename, file_size, folder_id, type)
    #         # Nếu result =1: thỏa mãn yêu cầu về total_size <50 và file_size <20
    #         if result == 1:
    #             # Lưu file vào folder
    #             folder_path = f"{USER_DOCUMENT}/{user_id}/{folder_id}"
    #             os.makedirs(folder_path, exist_ok=True)
    #
    #             with open(f"{folder_path}/{file.filename}", "wb") as buff:
    #                 shutil.copyfileobj(file.file, buff)
    #             # Chunking document
    #             if semantic_chunking:
    #                 chunks = self.semantic_chunking(f"{folder_path}/{file.filename}")
    #             else:
    #                 chunks = self.recursive_chunking(f"{folder_path}/{file.filename}")
    #             # bỏ vào vectorstore mới
    #             try:
    #                 new_db_for_user = self.create_vectorstore(chunks)
    #                 try:
    #                     # Nếu đã có db, hợp nhất db cũ với db mới
    #                     db_user = self.db
    #                     merged_db_user = self.merge_to_vectorstore(db_user, new_db_for_user, user_id, folder_id)
    #                     # return merged_db_user
    #                 except:  # Nếu chưa có db
    #                     new_db_for_user.save_local(f'{VECTOR_DATABASE}/{user_id}/{folder_id}')
    #                     # return new_db_for_user
    #                 return f"Successfully uploaded {file.filename}, num_splits: {len(chunks)}"
    #             except:
    #                 # sql_conn.delete_file(file.filename, user_id)
    #                 return "Incorrect API key provided, please make sure your API key is correct!"
    #         else:
    #             # sql_conn.delete_file(file.filename, user_id)
    #             return """Failed to upload document, the total size limit is 50Mb and the file size limit is 20Mb.
    #                     You can delete existed document to upload an other one!"""
    #     else:
    #         return "Only pdf, docx files are supported"
