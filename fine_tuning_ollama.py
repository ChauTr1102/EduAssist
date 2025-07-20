import os
import json
import argparse
import ollama
from datasets import load_dataset
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLLMSummarizer:
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
    
    def generate_summary(self, prompt, temperature=0.7, max_length=50):
        """Generate summary using Ollama model"""
        try:
            # Add length constraint to the prompt
            length_prompt = f"{prompt}\n\nHãy giữ tóm tắt ngắn gọn, không quá {max_length} ký tự."
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': length_prompt
                    }
                ],
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': max_length * 2  # Allow some buffer for token to character conversion
                }
            )
            generated_text = response['message']['content']
            
            # Truncate if still too long
            if len(generated_text) > max_length:
                generated_text = generated_text[:max_length] + "..."
                
            return generated_text
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3.2:3b")
    parser.add_argument("--output_file", type=str, default="./ollama_evaluation_results.json")
    parser.add_argument("--dataset_name", type=str, default="OpenHust/vietnamese-summarization")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_summary_length", type=int, default=50)
    return parser.parse_args()

def generate_summary_with_ollama(summarizer, prompt, temperature=0.7, max_length=50):
    """Generate summary using Ollama model"""
    return summarizer.generate_summary(prompt, temperature, max_length)

def calculate_summary_stats(original_summary, generated_summary, document):
    """Calculate statistics for summaries"""
    stats = {
        'document_length': len(document),
        'document_words': len(document.split()),
        'original_summary_length': len(original_summary),
        'original_summary_words': len(original_summary.split()),
        'generated_summary_length': len(generated_summary),
        'generated_summary_words': len(generated_summary.split()),
        'compression_ratio_original': len(document) / len(original_summary) if len(original_summary) > 0 else 0,
        'compression_ratio_generated': len(document) / len(generated_summary) if len(generated_summary) > 0 else 0,
    }
    return stats

def evaluate_model_on_dataset(summarizer, dataset, max_samples, temperature, max_length):
    """Evaluate Ollama model on Vietnamese summarization dataset"""
    results = []
    total_stats = {
        'total_samples': 0,
        'avg_document_length': 0,
        'avg_original_summary_length': 0,
        'avg_generated_summary_length': 0,
        'avg_compression_ratio_original': 0,
        'avg_compression_ratio_generated': 0,
    }
    
    # Take a subset of the data for evaluation
    eval_data = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    
    print(f"Evaluating {summarizer.model_name} on {len(eval_data)} samples...")
    
    for i, example in enumerate(tqdm(eval_data)):
        document = example['Document']
        original_summary = example['Summary']
        
        # Create prompt for summarization
        prompt = f"Tóm tắt văn bản sau đây một cách ngắn gọn và chính xác:\n\n{document}\n\nTóm tắt:"
        
        # Generate summary with Ollama
        generated_summary = generate_summary_with_ollama(summarizer, prompt, temperature, max_length)
        
        # Calculate statistics
        stats = calculate_summary_stats(original_summary, generated_summary, document)
        
        result = {
            'index': i,
            'document': document,
            'original_summary': original_summary,
            'generated_summary': generated_summary,
            'prompt': prompt,
            'stats': stats
        }
        results.append(result)
        
        # Update running averages
        total_stats['total_samples'] = i + 1
        total_stats['avg_document_length'] = (total_stats['avg_document_length'] * i + stats['document_length']) / (i + 1)
        total_stats['avg_original_summary_length'] = (total_stats['avg_original_summary_length'] * i + stats['original_summary_length']) / (i + 1)
        total_stats['avg_generated_summary_length'] = (total_stats['avg_generated_summary_length'] * i + stats['generated_summary_length']) / (i + 1)
        total_stats['avg_compression_ratio_original'] = (total_stats['avg_compression_ratio_original'] * i + stats['compression_ratio_original']) / (i + 1)
        total_stats['avg_compression_ratio_generated'] = (total_stats['avg_compression_ratio_generated'] * i + stats['compression_ratio_generated']) / (i + 1)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(eval_data)} samples")
            print(f"  Avg document length: {total_stats['avg_document_length']:.0f} chars")
            print(f"  Avg original summary: {total_stats['avg_original_summary_length']:.0f} chars")
            print(f"  Avg generated summary: {total_stats['avg_generated_summary_length']:.0f} chars")
    
    return results, total_stats

def save_results(results, total_stats, output_file):
    """Save evaluation results to JSON file"""
    output_data = {
        'summary_statistics': total_stats,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

def print_detailed_stats(total_stats):
    """Print detailed statistics about the evaluation"""
    print("\n" + "="*60)
    print("DETAILED EVALUATION STATISTICS")
    print("="*60)
    print(f"Total samples processed: {total_stats['total_samples']}")
    print(f"Average document length: {total_stats['avg_document_length']:.0f} characters")
    print(f"Average original summary length: {total_stats['avg_original_summary_length']:.0f} characters")
    print(f"Average generated summary length: {total_stats['avg_generated_summary_length']:.0f} characters")
    print(f"Average compression ratio (original): {total_stats['avg_compression_ratio_original']:.1f}x")
    print(f"Average compression ratio (generated): {total_stats['avg_compression_ratio_generated']:.1f}x")
    print("="*60)

def main():
    args = parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    print(f"Dataset loaded. Total samples: {len(dataset['train'])}")
    
    # Initialize Ollama summarizer
    try:
        summarizer = OllamaLLMSummarizer(args.model_name)
        print(f"Successfully initialized Ollama summarizer with model: {args.model_name}")
    except Exception as e:
        print(f"Failed to initialize Ollama summarizer: {e}")
        return
    
    # Evaluate model on dataset
    results, total_stats = evaluate_model_on_dataset(
        summarizer, 
        dataset, 
        args.max_samples, 
        args.temperature,
        args.max_summary_length
    )
    
    # Print detailed statistics
    print_detailed_stats(total_stats)
    
    # Save results
    save_results(results, total_stats, args.output_file)
    
    # Print some sample results
    print("\n" + "="*50)
    print("SAMPLE RESULTS:")
    print("="*50)
    
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"\n--- Sample {i+1} ---")
        print(f"Document (first 200 chars): {result['document'][:200]}...")
        print(f"\nOriginal Summary ({result['stats']['original_summary_length']} chars): {result['original_summary']}")
        print(f"\nGenerated Summary ({result['stats']['generated_summary_length']} chars): {result['generated_summary']}")
        print(f"\nStats:")
        print(f"  Document: {result['stats']['document_length']} chars, {result['stats']['document_words']} words")
        print(f"  Original compression: {result['stats']['compression_ratio_original']:.1f}x")
        print(f"  Generated compression: {result['stats']['compression_ratio_generated']:.1f}x")
        print("-" * 30)

if __name__ == "__main__":
    main()