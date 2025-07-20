import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import ollama
import requests
import json
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoBERTTranslator:
    """
    PhoBERT-based translator for Vietnamese <-> English
    Using multilingual models that support bidirectional translation
    """
    
    def __init__(self, model_name: str = "VietAI/envit5-translation"):
        """
        Initialize the translator with Vietnamese-English translation models
        
        Args:
            model_name: Hugging Face model name for translation
        """
        self.model_name = model_name
        self.model_type = "unknown"
        self.vi_to_en_model = None
        self.en_to_vi_model = None
        self.vi_to_en_tokenizer = None
        self.en_to_vi_tokenizer = None
        
        # Try bidirectional models first (can handle both directions)
        bidirectional_models = [
            ("facebook/mbart-large-50-many-to-many-mmt", "mbart"),  # Multilingual model
            ("VietAI/envit5-translation", "envit5"),  # Original model
            ("google/mt5-small", "mt5"),  # Backup multilingual model
        ]
        
        # Try unidirectional models (separate models for each direction)
        unidirectional_models = [
            ("Helsinki-NLP/opus-mt-vi-en", "Helsinki-NLP/opus-mt-en-vi", "opus"),  # Vi->En and En->Vi
        ]
        
        # First, try bidirectional models
        for model_name_try, model_type in bidirectional_models:
            try:
                logger.info(f"Attempting to load bidirectional model {model_name_try}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name_try)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name_try)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(self.device)
                
                # Use the same model for both directions
                self.vi_to_en_model = model
                self.en_to_vi_model = model
                self.vi_to_en_tokenizer = tokenizer
                self.en_to_vi_tokenizer = tokenizer
                self.model_name = model_name_try
                self.model_type = model_type
                logger.info(f"Successfully loaded bidirectional model: {model_name_try} (type: {model_type})")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name_try}: {e}")
                continue
        
        # If bidirectional models fail, try unidirectional models
        for vi_en_model, en_vi_model, model_type in unidirectional_models:
            try:
                logger.info(f"Attempting to load unidirectional models {vi_en_model} and {en_vi_model}...")
                
                # Load Vi->En model
                vi_to_en_tokenizer = AutoTokenizer.from_pretrained(vi_en_model)
                vi_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(vi_en_model)
                
                # Load En->Vi model
                en_to_vi_tokenizer = AutoTokenizer.from_pretrained(en_vi_model)
                en_to_vi_model = AutoModelForSeq2SeqLM.from_pretrained(en_vi_model)
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                vi_to_en_model.to(self.device)
                en_to_vi_model.to(self.device)
                
                self.vi_to_en_model = vi_to_en_model
                self.en_to_vi_model = en_to_vi_model
                self.vi_to_en_tokenizer = vi_to_en_tokenizer
                self.en_to_vi_tokenizer = en_to_vi_tokenizer
                self.model_name = f"{vi_en_model} + {en_vi_model}"
                self.model_type = model_type
                logger.info(f"Successfully loaded unidirectional models: {self.model_name} (type: {model_type})")
                return
            except Exception as e:
                logger.warning(f"Failed to load unidirectional models {vi_en_model}/{en_vi_model}: {e}")
                continue
        
        raise RuntimeError("Failed to load any translation model")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text for better translation
        
        Args:
            text: Raw Vietnamese text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove repeated spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove question marks that appear alone (artifacts from speech)
        text = re.sub(r'\s+\?\s+', ' ', text)
        
        # Clean up common speech-to-text artifacts
        text = re.sub(r'\bơi\b', '', text)  # Remove filler words
        text = re.sub(r'\bem ạ\b', '', text)  # Remove polite particles that might confuse translation
        text = re.sub(r'\banh ơi\b', '', text)  # Remove address terms
        
        # Normalize punctuation
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)  # Remove repeated punctuation
        
        # Remove very short fragments that might confuse the model
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        
        # Join sentences back
        result = '. '.join(sentences)
        
        # Ensure it ends with proper punctuation
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result
    
    def _chunk_text(self, text: str, max_length: int = 200) -> list:
        """
        Split text into smaller chunks for better translation
        
        Args:
            text: Text to chunk
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max_length, start a new chunk
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
            else:
                current_chunk += sentence + "."
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def translate_vi_to_en(self, text: str) -> str:
        """
        Translate Vietnamese text to English
        
        Args:
            text: Vietnamese text to translate
            
        Returns:
            English translation
        """
        try:
            # Preprocess the text
            text = self._preprocess_text(text)
            
            # If text is too long, chunk it
            if len(text) > 400:
                chunks = self._chunk_text(text, max_length=300)
                translations = []
                
                for chunk in chunks:
                    chunk_translation = self._translate_chunk_vi_to_en(chunk)
                    translations.append(chunk_translation)
                
                return ' '.join(translations)
            else:
                return self._translate_chunk_vi_to_en(text)
                
        except Exception as e:
            logger.error(f"Translation VI->EN failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_chunk_vi_to_en(self, text: str) -> str:
        """
        Translate a single chunk of Vietnamese text to English
        
        Args:
            text: Vietnamese text chunk to translate
            
        Returns:
            English translation
        """
        try:
            # Use the appropriate model and tokenizer for Vi->En
            model = self.vi_to_en_model
            tokenizer = self.vi_to_en_tokenizer
            
            # Prepare input based on model type
            if self.model_type == "opus":
                # OPUS models expect just the raw text
                input_text = text
            elif self.model_type == "mbart":
                # mBART models expect language codes
                input_text = text
                # Set source language for mBART
                tokenizer.src_lang = "vi_VN"
            elif self.model_type == "envit5":
                # EnviT5 models expect "vi: " prefix
                input_text = f"vi: {text}"
            elif self.model_type == "mt5":
                # MT5 models can use task prefix
                input_text = f"translate Vietnamese to English: {text}"
            else:
                # Default format
                input_text = text
            
            # Tokenize with appropriate settings
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation with better parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    min_length=10,  # Ensure minimum output length
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    forced_bos_token_id=tokenizer.lang_code_to_id.get("en_XX") if self.model_type == "mbart" else None
                )
            
            # Decode the output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output based on model type
            if self.model_type == "envit5" and translation.startswith("en:"):
                translation = translation[3:].strip()
            elif self.model_type == "mbart":
                # mBART sometimes includes language tokens, remove them
                translation = translation.replace("en_XX", "").replace("vi_VN", "").strip()
            
            # Additional cleanup
            translation = translation.replace("vi:", "").replace("en:", "").strip()
            
            # Validate translation quality
            if self._is_bad_translation(translation, text):
                logger.warning("Detected poor translation quality, trying alternative approach")
                return self._fallback_translation(text)
            
            logger.info(f"Translated VI->EN: {text[:50]}... -> {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation VI->EN failed: {e}")
            return self._fallback_translation(text)
    
    def _is_bad_translation(self, translation: str, original: str) -> bool:
        """
        Check if translation quality is poor
        
        Args:
            translation: Generated translation
            original: Original text
            
        Returns:
            True if translation is likely bad
        """
        # Check for repetitive patterns
        if len(translation) > 100:
            words = translation.split()[:10]  # Check first 10 words
            if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
                return True
        
        # Check for meaningless repetitive content
        repetitive_patterns = ["<i>", "(soft dramatic music)", "</i>", "♪", "music", "dramatic"]
        for pattern in repetitive_patterns:
            if translation.count(pattern) > 3:
                return True
        
        # Check if translation is much longer than reasonable
        if len(translation) > len(original) * 2:
            return True
        
        # Check if translation is too short compared to original
        if len(original) > 100 and len(translation) < 20:
            return True
        
        return False
    
    def _fallback_translation(self, text: str) -> str:
        """
        Fallback translation using simpler approach
        
        Args:
            text: Text to translate
            
        Returns:
            Simple translation or summary
        """
        try:
            # Use the appropriate model for the direction
            model = self.vi_to_en_model
            tokenizer = self.vi_to_en_tokenizer
            
            # Try with simpler input format
            inputs = tokenizer(
                text[:200],  # Use only first 200 chars
                return_tensors="pt",
                max_length=200,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.8
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = translation.replace("vi:", "").replace("en:", "").strip()
            
            if not self._is_bad_translation(translation, text):
                return translation
        except:
            pass
        
        # Last resort: return a summary of the original text
        sentences = text.split('.')[:3]  # Take first 3 sentences
        return "Vietnamese text about: " + '. '.join(sentences)[:100] + "..."
    
    def translate_en_to_vi(self, text: str) -> str:
        """
        Translate English text to Vietnamese
        
        Args:
            text: English text to translate
            
        Returns:
            Vietnamese translation
        """
        try:
            # Preprocess the text
            text = text.strip()
            
            # If text is too long, chunk it
            if len(text) > 400:
                sentences = text.split('.')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) > 300 and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + "."
                    else:
                        current_chunk += sentence + "."
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                translations = []
                for chunk in chunks:
                    chunk_translation = self._translate_chunk_en_to_vi(chunk)
                    translations.append(chunk_translation)
                
                return ' '.join(translations)
            else:
                return self._translate_chunk_en_to_vi(text)
                
        except Exception as e:
            logger.error(f"Translation EN->VI failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_chunk_en_to_vi(self, text: str) -> str:
        """
        Translate a single chunk of English text to Vietnamese
        
        Args:
            text: English text chunk to translate
            
        Returns:
            Vietnamese translation
        """
        try:
            # Use the appropriate model and tokenizer for En->Vi
            model = self.en_to_vi_model
            tokenizer = self.en_to_vi_tokenizer
            
            # Prepare input based on model type
            if self.model_type == "opus":
                # For OPUS models with separate models
                input_text = text
            elif self.model_type == "mbart":
                # mBART models expect language codes
                input_text = text
                # Set source language for mBART
                tokenizer.src_lang = "en_XX"
            elif self.model_type == "envit5":
                # EnviT5 models expect "en: " prefix for reverse translation
                input_text = f"en: {text}"
            elif self.model_type == "mt5":
                # MT5 models can use task prefix
                input_text = f"translate English to Vietnamese: {text}"
            else:
                # Default format
                input_text = text
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    min_length=5,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    forced_bos_token_id=tokenizer.lang_code_to_id.get("vi_VN") if self.model_type == "mbart" else None
                )
            
            # Decode the output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output based on model type
            if self.model_type == "envit5" and translation.startswith("vi:"):
                translation = translation[3:].strip()
            elif self.model_type == "mbart":
                translation = translation.replace("en_XX", "").replace("vi_VN", "").strip()
            elif self.model_type == "mt5":
                # Clean up MT5 output
                translation = translation.replace("translate English to Vietnamese:", "").strip()
            
            # Additional cleanup
            translation = translation.replace("vi:", "").replace("en:", "").strip()
            
            # Validate that we got a Vietnamese translation
            if self._is_english_text(translation):
                logger.warning("Translation resulted in English text, attempting fallback")
                return self._fallback_en_to_vi_translation(text)
            
            logger.info(f"Translated EN->VI: {text[:50]}... -> {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation EN->VI failed: {e}")
            return self._fallback_en_to_vi_translation(text)
    
    def _is_english_text(self, text: str) -> bool:
        """
        Check if text is primarily English (indicating failed Vietnamese translation)
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be English
        """
        # Common English words that shouldn't appear in Vietnamese
        english_indicators = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would', 
                            'this', 'that', 'with', 'from', 'they', 'their', 'there', 'where', 'when']
        
        words = text.lower().split()[:10]  # Check first 10 words
        english_word_count = sum(1 for word in words if word in english_indicators)
        
        # If more than 30% of words are common English words, it's likely English
        if len(words) > 0 and english_word_count / len(words) > 0.3:
            return True
        
        return False
    
    def _fallback_en_to_vi_translation(self, text: str) -> str:
        """
        Fallback translation for EN->VI using alternative approach
        
        Args:
            text: English text to translate
            
        Returns:
            Vietnamese translation or description
        """
        try:
            # Try with simpler approach
            model = self.en_to_vi_model
            tokenizer = self.en_to_vi_tokenizer
            
            # Use a very simple input format
            simple_text = text[:100]  # Even shorter text
            inputs = tokenizer(
                simple_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = translation.replace("vi:", "").replace("en:", "").strip()
            
            if not self._is_english_text(translation) and len(translation) > 10:
                return translation
        except:
            pass
        
        # Last resort: create a Vietnamese description
        return f"Bản tóm tắt về: {text[:50]}..."

class LocalLLMSummarizer:
    """
    Local LLM summarizer using Ollama
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the local LLM summarizer
        
        Args:
            model_name: Ollama model name for summarization
        """
        self.model_name = model_name
        
        # Initialize Ollama client
        try:
            self.client = ollama.Client()
            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
        
        # Check if model is available
        self._check_and_pull_model()
    
    def _check_and_pull_model(self):
        """Check if model is available and pull if necessary"""
        try:
            models_response = self.client.list()
            available_models = []
            
            # Handle different response formats
            if 'models' in models_response:
                for model in models_response['models']:
                    # Try different possible keys
                    model_name_key = model.get('name') or model.get('model') or model.get('id')
                    if model_name_key:
                        available_models.append(model_name_key)
            
            logger.info(f"Available models: {available_models}")
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                logger.info(f"Attempting to pull model {self.model_name}...")
                try:
                    self.client.pull(self.model_name)
                    logger.info(f"Successfully pulled model {self.model_name}")
                except Exception as pull_error:
                    logger.error(f"Failed to pull model {self.model_name}: {pull_error}")
                    raise
            
            logger.info(f"Using local LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to check/pull model: {e}")
            logger.info("Proceeding without model verification - will try to use the model directly")
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and model is accessible"""
        try:
            # Try a simple generation to test connectivity
            response = self.client.generate(
                model=self.model_name,
                prompt="Hello",
                options={'num_predict': 1}
            )
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def summarize(self, text: str, max_length: int = 50) -> str:
        """
        Summarize text using local LLM
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        try:
            prompt = f"""
            Please provide a concise summary of the following text. 
            Keep the summary under {max_length} words and focus on the main points.
            
            Text to summarize:
            {text}
            
            Summary:
            """
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': max_length * 2,  # Use num_predict instead of max_tokens
                    'top_p': 0.9
                }
            )
            
            summary = response['response'].strip()
            logger.info(f"Summarized text: {len(text)} chars -> {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text

class VietnameseSummarizationPipeline:
    """
    Complete pipeline for Vietnamese text summarization using PhoBERT and local LLM
    """
    
    def __init__(self, 
                 translation_model: str = "VietAI/envit5-translation",
                 llm_model: str = "llama3.2:3b"):
        """
        Initialize the complete pipeline
        
        Args:
            translation_model: PhoBERT-based translation model
            llm_model: Local LLM model for summarization
        """
        logger.info("Initializing Vietnamese Summarization Pipeline...")
        
        # Initialize translator
        try:
            self.translator = PhoBERTTranslator(translation_model)
            logger.info("Translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translator: {e}")
            raise
        
        # Initialize summarizer
        try:
            self.summarizer = LocalLLMSummarizer(llm_model)
            logger.info("Summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {e}")
            raise
            
        # Test the connection
        if not self.summarizer.test_connection():
            logger.warning("Ollama connection test failed. Make sure Ollama is running.")
            logger.info("You can start Ollama with: ollama serve")
        
        logger.info("Vietnamese Summarization Pipeline initialized successfully")
    
    def process(self, vietnamese_text: str, summary_length: int = 50) -> Dict[str, str]:
        """
        Process Vietnamese text through the complete pipeline
        
        Args:
            vietnamese_text: Input Vietnamese text
            summary_length: Desired summary length in words
            
        Returns:
            Dictionary containing all intermediate and final results
        """
        logger.info("Starting Vietnamese text summarization pipeline...")
        
        # Step 1: Translate Vietnamese to English
        logger.info("Step 1: Translating Vietnamese to English")
        english_text = self.translator.translate_vi_to_en(vietnamese_text)
        
        # Step 2: Summarize English text using local LLM
        logger.info("Step 2: Summarizing English text")
        english_summary = self.summarizer.summarize(english_text, summary_length)
        
        # Step 3: Translate English summary back to Vietnamese
        logger.info("Step 3: Translating summary back to Vietnamese")
        vietnamese_summary = self.translator.translate_en_to_vi(english_summary)
        
        results = {
            'original_vietnamese': vietnamese_text,
            'english_translation': english_text,
            'english_summary': english_summary,
            'vietnamese_summary': vietnamese_summary
        }
        
        logger.info("Pipeline completed successfully!")
        return results
    
    def summarize_vietnamese(self, text: str, summary_length: int = 200) -> str:
        """
        Simplified method that returns only the final Vietnamese summary
        
        Args:
            text: Vietnamese text to summarize
            summary_length: Desired summary length
            
        Returns:
            Vietnamese summary
        """
        results = self.process(text, summary_length)
        return results['vietnamese_summary']

# Example usage and testing
def main():
    """
    Example usage of the Vietnamese Summarization Pipeline
    """
    # Sample Vietnamese text
    sample_text = """
    Đây chính là bàn học của một thủ khoa 3 điểm 10  
    Mặc dù là một góc học tập nhỏ bé và kiêm tốn  
    Nhưng mà đã tạo ra một thủ khoa của khối A00 với 3 con 10  
    Điểm số tuyệt đối luôn anh ơi  
    Trời ơi, anh thấy xấu hổ quá  
    Thôi thì bây giờ mời nhân vật chính của chúng ta ngày hôm nay vào  
    Minh ơi, chào Minh  
    Minh ơi, thế chắc bây giờ là chị nhà Minh có thể là  
    Giới thiệu cái góc học tập quen thuộc này của em cho các bạn khán giả xem được không?  
    Góc học tập quen thuộc của em thì cũng khá nhỏ bé  
    Thường thì ở đây có mặt bàn thì chỉ vừa đủ để 1-2 cuốn sách trong lúc mình làm bài thôi  
    Và để ở đây là những cuốn mà mình cần thiết để hôm sau mang đi học  
    Về cảm xúc của em bây giờ thì như nào?  
    Sau khi hôm qua biết là mình là một thủ khoa 3 điểm 10  
    Thành thật mà nói thì khi biết được mình  
    Trong khi tra điểm thì em đã có cảm giác đoán trước rồi  
    Đoán trước rồi  
    Thế nhưng mà đến lúc điểm trao về tay
    Cảm giác cực kỳ vui rồi  
    Thật  
    Vỗ hoàng không?  
    Cũng cực kỳ vỗ hoàng  
    Đây chỉ là một giấc mơ  
    Nhưng mà giấc mơ này có thật  
    Giấc mơ này có thật em ạ  
    Bởi vì là để đạt được 3 điểm 10 năm nay  
    Hay là kể cả những năm về trước  
    Không phải là một điều đơn giản cả  
    Bởi vì năm nay thì hình thức đề cũng mới hơn rất nhiều  
    Và anh cũng muốn hỏi thêm là  
    Lúc thi xong thì em có tự tin là mình sẽ đạt được 3 điểm 10 như này không?  Thành thật mà nói thì ký thi này nó phụ trọng vào rất nhiều ước đối như là  Tô và các mạng đề có kiểu  Lúc check đáp án thì em biết là  Mình đoán trước là em 30 rồi  Nhưng mà em đang lo rất nhiều ước đối  Rằng sau như là tô sai hoặc tô mờ  Khiến cho bài nó lệch so với dự đoán ban đầu  Thế sau khi mà mình viết điểm rồi  Xong rồi đến khi mà nhận được kết quả chính thức  Là mình thủ khoa với A00  Thì có khoe gia đình không?  Xong rồi gia đình như thế nào em?  Sau khi em đạt được cái điểm thủ khoa này  Thì lần đầu tiên thì em cũng phải khoe về gia đình đầu tiên  Vậy là em đã đạt được cái điểm thủ khoa này  Mọi người cũng rất vui và ước kích  Bởi vì lúc này là lần đầu  Bình thường là em có dành nhiều thời gian để học không?  Một ngày em sẽ dành khoảng bao nhiêu thời gian để học?
    """
    
    try:
        # Initialize pipeline
        pipeline = VietnameseSummarizationPipeline()
        
        # Process the text
        results = pipeline.process(sample_text, summary_length=30)
        
        # Print results
        print("=== VIETNAMESE SUMMARIZATION PIPELINE RESULTS ===")
        print(f"\nOriginal Vietnamese Text:")
        print(results['original_vietnamese'])
        print(f"\nEnglish Translation:")
        print(results['english_translation'])
        print(f"\nEnglish Summary:")
        print(results['english_summary'])
        print(f"\nVietnamese Summary:")
        print(results['vietnamese_summary'])
        
        # Test simplified method
        # print("\n=== SIMPLIFIED METHOD ===")
        # simple_summary = pipeline.summarize_vietnamese(sample_text, 80)
        # print(f"Simple Vietnamese Summary: {simple_summary}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    # Install required packages
    print("Make sure you have installed the required packages:")
    print("pip install torch transformers ollama requests")
    print("\nMake sure Ollama is running:")
    print("ollama serve")
    print("ollama pull llama3.2:3b")
    print("\nIf you're getting connection errors, try:")
    print("1. Check if Ollama is running: ollama list")
    print("2. Start Ollama service: ollama serve")
    print("3. Pull the model: ollama pull llama3.2:3b")
    print("\n" + "="*50 + "\n")
    main()