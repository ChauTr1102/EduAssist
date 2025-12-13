"""
Ollama MapReduce Pipeline - All-in-one implementation
Gộp toàn bộ Generator, Pipeline và Ollama client vào một file duy nhất
GIỮ ĐÚNG LOGIC GỐC từ Generator.py, pipeline.py, utils.py
"""

import copy
import re
import yaml
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Install with: pip install ollama")


# ============= OLLAMA CLIENT WRAPPER =============

class OllamaClient:
    """Wrapper đơn giản cho Ollama API"""
    
    def __init__(self, model: str, host: str = "http://localhost:11434", 
                 temperature: float = 0.7, max_tokens: int = 1024):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")
        
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client(host=host)
    
    def generate(self, prompt: str) -> str:
        """Generate response từ Ollama"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    "num_ctx": 7000
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "[ERROR]"
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate responses cho nhiều prompts (tuần tự)"""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"    Processing {i+1}/{len(prompts)}...")
            result = self.generate(prompt)
            results.append(result)
        return results


# ============= GENERATOR (LOGIC GỐC TỪ Generator.py) =============

class OllamaMapReduceGenerator:
    """
    Generator cho MapReduce pipeline với Ollama
    GIỮ ĐÚNG LOGIC GỐC từ Generator.py
    """
    
    def __init__(self, ollama_client: OllamaClient, tokenizer, config: Dict):
        self.client = ollama_client
        self.tokenizer = tokenizer
        self.config = config
        
        self.first_prompt = config['map_prompt']
        self.gen_args = config.get('gen_args', {})
    
    def get_prompt_length(self, prompt, **kwargs: Any) -> int:
        """Logic gốc từ Generator.get_prompt_length"""
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        return len(self.tokenizer.encode(prompt, **kwargs))
    
    def get_prompt_length_no_special(self, prompt, **kwargs: Any) -> int:
        """Logic gốc từ Generator.get_prompt_length_no_special"""
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False, **kwargs))
    
    def get_prompt_length_format(self, prompt, **kwargs: Any) -> int:
        """Logic gốc từ Generator.get_prompt_length_format"""
        if isinstance(prompt, list):
            prompt = ''.join(self.format_chunk_information(prompt))
        return len(self.tokenizer.encode(prompt, **kwargs))
    
    def join_docs(self, docs: List[str]) -> str:
        """Logic gốc từ Generator.join_docs"""
        if isinstance(docs, str):
            return docs
        return '\n\n'.join(docs)
    
    def format_chunk_information(self, docs):
        """Logic gốc từ Generator.format_chunk_information"""
        new_docs = [f'Information of Chunk {index}:\n{d}\n' for index, d in enumerate(docs)]
        return new_docs
    
    def split_sentences(self, text: str, spliter: str) -> List[str]:
        """Logic gốc từ Generator.split_sentences"""
        text = text.strip()
        sentence_list = re.split(spliter, text)
        
        if spliter != ' ':
            sentences = ["".join(i) for i in zip(sentence_list[0::2], sentence_list[1::2])]
            if len(sentence_list) % 2 != 0 and sentence_list[-1] != '':
                sentences.append(sentence_list[-1])
        else:
            sentences = [i+' ' for i in sentence_list if i != '']
            if sentences:
                sentences[-1] = sentences[-1].strip()
        return sentences
    
    def split_into_chunks(self, text, chunk_size, spliter=r'([。！？；.?!;])'):
        """Logic gốc từ Generator.split_into_chunks"""
        sentences = self.split_sentences(text, spliter)
        chunks = []
        current_chunk = ""
        
        for s_idx, sentence in enumerate(sentences):
            sentence_length = self.get_prompt_length(sentence)
            
            if self.get_prompt_length(current_chunk) + sentence_length <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    if self.get_prompt_length(current_chunk) <= chunk_size:
                        chunks.append(current_chunk)
                    else:
                        if spliter != ' ':  # Avoid infinite loops
                            chunks.extend(self.split_into_chunks(current_chunk, chunk_size=chunk_size, spliter=' '))
                current_chunk = sentence
        
        if current_chunk != '':
            if self.get_prompt_length(current_chunk) <= chunk_size:
                chunks.append(current_chunk)
            else:
                if spliter != ' ':  # Avoid infinite loops
                    chunks.extend(self.split_into_chunks(current_chunk, chunk_size=chunk_size, spliter=' '))
        
        # ===== LOGIC GỐC: Re-segment the last two blocks =====
        if len(chunks) > 1 and self.get_prompt_length(chunks[-1]) < chunk_size//2:
            last_chunk = chunks.pop()
            penultimate_chunk = chunks.pop()
            combined_text = penultimate_chunk + last_chunk
            
            new_sentences = self.split_sentences(combined_text, spliter)
            
            # Reallocate sentence using double pointer
            new_penultimate_chunk = ""
            new_last_chunk = ""
            i, j = 0, len(new_sentences) - 1
            
            while i <= j and len(new_sentences) != 1:
                flag = False
                if self.get_prompt_length(new_penultimate_chunk + new_sentences[i]) <= chunk_size:
                    flag = True
                    new_penultimate_chunk += new_sentences[i]
                    if i == j:
                        break  
                    i += 1
                if self.get_prompt_length(new_last_chunk + new_sentences[j]) <= chunk_size:
                    new_last_chunk = new_sentences[j] + new_last_chunk
                    j -= 1
                    flag = True
                if flag == False:
                    break
            if i < j:
                remaining_sentences = new_sentences[i:j+1]
                if remaining_sentences:
                    remaining_text = "".join(remaining_sentences)
                    words = remaining_text.split(' ')
                    end_index = len(words)-1
                    for index, w in enumerate(words):
                        if self.get_prompt_length(' '.join([new_penultimate_chunk, w])) <= chunk_size:
                            new_penultimate_chunk = ' '.join([new_penultimate_chunk, w])
                        else:
                            end_index = index
                            break
                    if end_index != len(words)-1:
                        new_last_chunk = ' '.join(words[end_index:]) + ' ' + new_last_chunk
            if len(new_sentences) == 1:
                chunks.append(penultimate_chunk)
                chunks.append(last_chunk)
            else:
                chunks.append(new_penultimate_chunk)
                chunks.append(new_last_chunk)
        
        return chunks
    
    def chunk_docs(self, doc: str, chunk_size: int, separator='\n', chunk_overlap=0, question=None) -> List[str]:
        """Logic gốc từ Generator.chunk_docs"""
        chunk_size = chunk_size - self.get_prompt_length(self.first_prompt) - self.gen_args.get('max_tokens', 300)
        if question is not None:
            chunk_size = chunk_size - self.get_prompt_length(question)
        
        splits = doc.split(separator)
        splits = [s for s in splits if s != '']
        separator_len = self.get_prompt_length_no_special(separator)
        
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self.get_prompt_length_no_special(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
                if total > chunk_size:
                    print(f"Created a chunk of size {total}, which is longer than the specified {chunk_size}")
                    
                    if len(current_doc) == 1:  # if one chunk is too long
                        split_again = self.split_into_chunks(current_doc[0], chunk_size)
                        docs.extend(split_again)
                        current_doc = []
                        total = 0
                
                if len(current_doc) > 0:
                    doc = separator.join(current_doc)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= self.get_prompt_length_no_special(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        # Check if the last one exceeds
        if current_doc:
            if len(current_doc) == 1 and self.get_prompt_length_no_special(current_doc[-1]) > chunk_size:
                split_again = self.split_into_chunks(current_doc[0], chunk_size)
                docs.extend(split_again)
                current_doc = []
            else:
                doc = separator.join(current_doc)
                if doc is not None:
                    docs.append(doc)
        docs = [d for d in docs if d.strip() != ""]
        return docs
    
    def split_list_of_docs(self, docs: List[str], length_func, token_max: int, **kwargs: Any) -> List[List[str]]:
        """Logic gốc từ utils.split_list_of_docs"""
        new_result_doc_list = []
        _sub_result_docs = []
        for doc in docs:
            _sub_result_docs.append(doc)
            _num_tokens = length_func(_sub_result_docs, **kwargs)
            if _num_tokens > token_max:
                if len(_sub_result_docs) == 1:
                    raise ValueError(
                        "A single document was longer than the context length,"
                        " we cannot handle this."
                    )
                new_result_doc_list.append(_sub_result_docs[:-1])
                _sub_result_docs = _sub_result_docs[-1:]
        new_result_doc_list.append(_sub_result_docs)
        return new_result_doc_list
    
    def mr_map(self, context: List[str], question) -> List[str]:
        """Logic gốc từ Generator.mr_map"""
        prompt = self.config['map_prompt']
        print("=====Map=====")
        
        prompts = []
        for i, item in enumerate(context):
            formatted_prompt = prompt.format(question=question, context=item)
            prompts.append(formatted_prompt)
        
        res = self.client.batch_generate(prompts)
        print(f'map result: {len(res)} items')
        return res
    
    def mr_collapse(self, docs: List[str], question: str, token_max: Optional[int] = None,
                    max_retries: Optional[int] = None) -> List[str]:
        """Logic gốc từ Generator.mr_collapse với while loop đệ quy"""
        result_docs = docs
        
        prompt = self.config['collapse_prompt']
        num_tokens = self.get_prompt_length_format(result_docs)
        prompt_len = self.get_prompt_length(prompt)
        _token_max = token_max - prompt_len - self.gen_args.get('max_tokens', 300)
        retries: int = 0
        
        while num_tokens is not None and num_tokens > _token_max:
            new_result_doc_list = self.split_list_of_docs(result_docs, self.get_prompt_length_format, _token_max,)
            result_docs = []
            current_batch = []
            
            for index, docs in enumerate(new_result_doc_list):
                formatted_prompt = prompt.format(question=question, context=self.join_docs(docs))
                current_batch.append(formatted_prompt)
            
            result_docs = self.client.batch_generate(current_batch)
            
            num_tokens = self.get_prompt_length_format(result_docs)
            retries += 1
            if max_retries and retries == max_retries:
                raise ValueError(f"Exceed {max_retries} tries to collapse document to {_token_max} tokens.")
        
        print("=====Collapse=====")
        print(f"Collapsed to {len(result_docs)} items")
        return result_docs
    
    def mr_reduce(self, context: List[str], question):
        """Logic gốc từ Generator.mr_reduce"""
        prompt = self.config['reduce_prompt']
        context_formatted = ''.join(self.format_chunk_information(context))
        print("=====Reduce=====")
        
        formatted_prompt = prompt.format(context=context_formatted, question=question)
        result = self.client.generate(formatted_prompt)
        
        print(f"Reduce complete")
        return result


# ============= PIPELINE (LOGIC GỐC TỪ pipeline.py) =============

class OllamaMapReducePipeline:
    """
    Pipeline hoàn chỉnh cho MapReduce với Ollama
    GIỮ ĐÚNG LOGIC GỐC từ BasePipeline
    """
    
    def __init__(self, config: Dict, print_intermediate_path=None, doc_id=None):
        """
        Args:
            config: Dict chứa cấu hình:
                - ollama: {model, host, tokenizer}
                - gen_args: {temperature, max_tokens}
                - map_prompt, collapse_prompt, reduce_prompt
        """
        # Khởi tạo Ollama client
        ollama_config = config.get('ollama', {})
        gen_args = config.get('gen_args', {})

        self.ollama_client = OllamaClient(
            model=ollama_config.get('model', 'shmily_006/Qw3:4b_4bit'),
            host=ollama_config.get('host', 'http://localhost:11434'),
            temperature=gen_args.get('temperature', 0.5),
            max_tokens=gen_args.get('max_tokens', 1024)
        )
        
        # Khởi tạo tokenizer
        tokenizer_path = ollama_config.get('tokenizer', 'Qwen/Qwen3-4B')
        print(f"Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Khởi tạo generator với logic gốc
        self.generator = OllamaMapReduceGenerator(
            ollama_client=self.ollama_client,
            tokenizer=self.tokenizer,
            config=config
        )
        
        self.config = config
        self.print_intermediate_path = print_intermediate_path
        self.doc_id = doc_id
    
    def remove_chunk(self, chunks: list, irrelevant_note=['[NOT MENTIONED]'], question=''):
        """Logic gốc từ BasePipeline.remove_chunk"""
        new_chunks = []
        # If the topic is not mentioned
        for q in question:
            for note in irrelevant_note:
                if note.upper() in q.upper():
                    return chunks
        
        for chunk in chunks:
            flag = False
            for note in irrelevant_note:
                if note.upper() in chunk.upper():
                    flag = True
                    break
            if not flag:
                new_chunks.append(chunk)
        return new_chunks
    
    def run(self, doc: str, question: str, chunk_size: int = 2048) -> str:
        """
        Logic gốc từ BasePipeline.run
        
        Args:
            doc: Document text
            question: Question to answer
            chunk_size: Size of each chunk in tokens
            
        Returns:
            Final answer
        """
        doc_tokens = self.generator.get_prompt_length(doc)
        print(f"\n{'='*60}")
        print(f"📄 Document: {len(doc)} chars, {doc_tokens} tokens")
        print(f"❓ Question: {question}")
        print(f"🔧 Chunk size: {chunk_size} tokens")
        print(f"{'='*60}\n")
        
        # 1. Chunk document (logic gốc)
        split_docs = self.generator.chunk_docs(doc, chunk_size, question=question)
        contexts = split_docs
        print(f"✂️  Split into {len(split_docs)} chunks\n")
        
        # 2. Map stage (logic gốc)
        map_result = self.generator.mr_map(split_docs, question)
        map_result = self.remove_chunk(map_result, question=question, irrelevant_note=['[NO INFORMATION]'])
        
        # 3. Collapse stage (logic gốc)
        collapse_result = self.generator.mr_collapse(map_result, question, token_max=chunk_size)
        collapse_result = self.remove_chunk(collapse_result, question=question, irrelevant_note=['[NO INFORMATION]'])
        
        # 4. Reduce stage (logic gốc)
        reduce_result = self.generator.mr_reduce(collapse_result, question)
        
        print(f"\n{'='*60}")
        print("✅ Pipeline complete")
        print(f"{'='*60}\n")
        
        return reduce_result


# ============= HELPER FUNCTIONS =============

def load_config(config_path: str) -> Dict:
    """
    Load config từ file YAML
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Load config từ file YAML
    config = load_config("config_ollama_mapreduce.yaml")
    
    # Khởi tạo pipeline
    pipeline = OllamaMapReducePipeline(config)
    
    # Load document
    with open("/home/bojjoo/Code/EduAssist/test_data/hop_quochoi_lan10_khoaXV.txt") as f:
        document = f.read()
    
    # Chạy pipeline
    question = "Tóm tắt các ý chính của cuộc họp, trình bày rõ ràng thành từng mục nếu cần thiết"
    result = pipeline.run(document, question, chunk_size=4096)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result)