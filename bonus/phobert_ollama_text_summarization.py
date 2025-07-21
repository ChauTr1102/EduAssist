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
    Using mBERT or similar multilingual models fine-tuned for translation
    """
    
    def __init__(self, model_name: str = "VietAI/envit5-translation"):
        """
        Initialize the translator with a Vietnamese-English translation model
        
        Args:
            model_name: Hugging Face model name for translation
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"Loaded translation model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            # Fallback to a general multilingual model
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
    
    def translate_vi_to_en(self, text: str) -> str:
        """
        Translate Vietnamese text to English
        
        Args:
            text: Vietnamese text to translate
            
        Returns:
            English translation
        """
        try:
            # Prepare input with language codes
            input_text = f"vi: {text}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output (remove language prefix if present)
            if translation.startswith("en:"):
                translation = translation[3:].strip()
            
            logger.info(f"Translated VI->EN: {text[:50]}... -> {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation VI->EN failed: {e}")
            return text  # Return original text if translation fails
    
    def translate_en_to_vi(self, text: str) -> str:
        """
        Translate English text to Vietnamese
        
        Args:
            text: English text to translate
            
        Returns:
            Vietnamese translation
        """
        try:
            # Prepare input with language codes
            input_text = f"en: {text}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output (remove language prefix if present)
            if translation.startswith("vi:"):
                translation = translation[3:].strip()
            
            logger.info(f"Translated EN->VI: {text[:50]}... -> {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation EN->VI failed: {e}")
            return text  # Return original text if translation fails

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
    Trí tuệ nhân tạo (AI) đang phát triển nhanh chóng và có tác động sâu rộng đến nhiều lĩnh vực của cuộc sống. 
    Công nghệ AI đã được ứng dụng trong y tế, giáo dục, giao thông, tài chính và nhiều ngành khác. 
    Trong y tế, AI giúp chẩn đoán bệnh chính xác hơn và phát triển thuốc mới. 
    Trong giáo dục, AI cá nhân hóa quá trình học tập cho từng học sinh. 
    Tuy nhiên, sự phát triển của AI cũng đặt ra nhiều thách thức về đạo đức, an toàn và tác động đến việc làm. 
    Chúng ta cần có những quy định và chính sách phù hợp để đảm bảo AI phát triển một cách có trách nhiệm 
    và mang lại lợi ích cho toàn xã hội.
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