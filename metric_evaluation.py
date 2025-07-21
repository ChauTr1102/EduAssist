import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import GroupKFold
import evaluate
import json
import numpy as np
from datetime import datetime
from EduAssist.phobert_ollama_text_summarization import VietnameseSummarizationPipeline  # import your pipeline

# Check MoverScore availability
MOVERSCORE_AVAILABLE = False
try:
    from moverscore import get_idf_dict, word_mover_score
    MOVERSCORE_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"Warning: MoverScore not available due to dependency issues: {e}")
    print("Continuing without MoverScore metric...")

# ——— Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1) METRICS SETUP
bleu_metric   = evaluate.load("sacrebleu")
chrf_metric   = evaluate.load("chrf")
bertscore     = evaluate.load("bertscore")
rouge_metric  = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
# drop evaluate.load("mover_score") entirely

def compute_summarization_metrics(hyps: List[str], refs: List[str]) -> Dict:
    # ROUGE/METEOR as before
    rouge_res  = rouge_metric.compute(predictions=hyps, references=refs, use_stemmer=True)
    meteor_res = meteor_metric.compute(predictions=hyps, references=refs)["meteor"]

    # Handle different ROUGE return formats
    if hasattr(rouge_res["rouge1"], 'mid'):
        # Old format with .mid attribute
        rouge1_score = rouge_res["rouge1"].mid.fmeasure
        rouge2_score = rouge_res["rouge2"].mid.fmeasure
        rougeL_score = rouge_res["rougeL"].mid.fmeasure
    else:
        # New format with direct float values
        rouge1_score = rouge_res["rouge1"]
        rouge2_score = rouge_res["rouge2"]
        rougeL_score = rouge_res["rougeL"]

    results = {
        "ROUGE-1":  rouge1_score,
        "ROUGE-2":  rouge2_score,
        "ROUGE-L":  rougeL_score,
        "METEOR":   meteor_res,
    }

    # ----- MoverScore (only if available) -----
    if MOVERSCORE_AVAILABLE:
        try:
            # Build idf dicts for reference and hypothesis
            idf_ref = get_idf_dict(refs)  # maps token→idf
            idf_hyp = get_idf_dict(hyps)
            # Compute sentence-level scores
            mover_scores = word_mover_score(
                refs,       # list of reference strings
                hyps,       # list of hypothesis strings
                idf_ref,
                idf_hyp,
                stop_words=[],    # you can pass a stop-words list if desired
                n_gram=1,
                remove_subwords=True
            )
            # Average over the batch
            mover_mean = sum(mover_scores) / len(mover_scores)
            results["MoverScore"] = mover_mean
        except Exception as e:
            print(f"Warning: MoverScore computation failed: {e}")

    return results

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_results_to_json(results: Dict, filename: str = None):
    """Save evaluation results to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    # Convert numpy types to native Python types
    results_clean = convert_numpy_types(results)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        return None

# ——— Load Vietnamese news-summary dataset ———
# Note: Using a more common Vietnamese dataset that's available
try:
    # Try common Vietnamese datasets
    ds = load_dataset("csebuetnlp/xlsum", "vietnamese")
    train = ds["train"]
    val = ds["validation"]
    print(f"Loaded XL-Sum Vietnamese dataset: {len(train)} training examples, {len(val)} validation examples")
except Exception as e:
    print(f"Warning: Could not load XL-Sum Vietnamese dataset: {e}")
    try:
        # Fallback to a smaller sample dataset for testing
        ds = load_dataset("cnn_dailymail", "3.0.0")
        # Take a small subset for testing
        train = ds["train"].select(range(1000))
        val = ds["validation"].select(range(100))
        print(f"Loaded CNN/DailyMail dataset subset for testing: {len(train)} training examples, {len(val)} validation examples")
        print("Note: This is English data for testing purposes only")
    except Exception as e2:
        print(f"Error loading any dataset: {e2}")
        print("Please ensure you have a valid Vietnamese summarization dataset")
        exit(1)

# ——— Initialize your pipeline ———
pipeline = VietnameseSummarizationPipeline(
    translation_model="VietAI/envit5-translation",
    llm_model="llama3.2:3b"
)

# ——— k-Fold Cross-Validation ———
def cross_validate(n_splits=5):
    # Handle different dataset schemas
    if "text" in train.column_names:
        texts = train["text"]
        refs = train["summary"] if "summary" in train.column_names else train["highlights"]
    elif "article" in train.column_names:
        texts = train["article"]
        refs = train["highlights"] if "highlights" in train.column_names else train["summary"]
    else:
        print(f"Available columns: {train.column_names}")
        raise ValueError("Could not find text/article column in dataset")
    
    groups = list(range(len(texts)))  # ensure each doc stays together

    gkf = GroupKFold(n_splits=n_splits)
    all_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(texts, groups=groups), 1):
        logger.info(f"Starting fold {fold}/{n_splits}…")

        # collect train/val splits using dataset.select() to avoid indexing issues
        train_subset = train.select(train_idx)
        val_subset = train.select(val_idx)
        
        # Extract texts and references using the same column logic
        if "text" in train.column_names:
            tr_texts = train_subset["text"]
            va_texts = val_subset["text"]
            va_refs = val_subset["summary"] if "summary" in train.column_names else val_subset["highlights"]
        elif "article" in train.column_names:
            tr_texts = train_subset["article"]
            va_texts = val_subset["article"]
            va_refs = val_subset["highlights"] if "highlights" in train.column_names else val_subset["summary"]

        # (Optionally) you could fine-tune pipeline here on tr_texts/tr_refs via your own logic.
        # For now we just evaluate zero-shot on va_texts.

        hyps = []
        for txt in va_texts:
            hyp = pipeline.summarize_vietnamese(txt, summary_length=50)
            hyps.append(hyp)

        scores = compute_summarization_metrics(hyps, va_refs)
        logger.info(f"Fold {fold} metrics: {scores}")
        all_scores.append(scores)

    # average across folds
    avg = {
        metric: sum(d[metric] for d in all_scores) / n_splits
        for metric in all_scores[0]
    }
    return avg

if __name__ == "__main__":
    # Store all results for saving
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "translation_model": "VietAI/envit5-translation",
            "llm_model": "llama3.2:3b",
            "summary_length": 50
        },
        "dataset_info": {
            "name": "Unknown",
            "train_size": len(train),
            "val_size": len(val)
        }
    }
    
    # Add dataset information
    if hasattr(ds, 'config_name'):
        evaluation_results["dataset_info"]["name"] = f"{ds.builder_name}/{ds.config_name}" if ds.config_name else ds.builder_name
    
    # 1) Validate on held-out validation set
    logger.info("Evaluating on official validation split…")
    
    # Handle different dataset schemas
    if "text" in val.column_names:
        val_texts = val["text"]
        val_refs = val["summary"] if "summary" in val.column_names else val["highlights"]
    elif "article" in val.column_names:
        val_texts = val["article"]
        val_refs = val["highlights"] if "highlights" in val.column_names else val["summary"]
    else:
        print(f"Available columns: {val.column_names}")
        raise ValueError("Could not find text/article column in validation dataset")
    
    hyps_val = [pipeline.summarize_vietnamese(t, summary_length=50) for t in val_texts]
    val_scores = compute_summarization_metrics(hyps_val, val_refs)
    print("Validation set metrics:", val_scores)
    
    # Add validation results
    evaluation_results["validation_metrics"] = val_scores

    # 2) k-Fold CV on training set
    logger.info("Starting k-fold cross-validation...")
    cv_scores = cross_validate(n_splits=5)
    print("5-fold CV metrics:", cv_scores)
    
    # Add cross-validation results
    evaluation_results["cross_validation"] = {
        "n_splits": 5,
        "average_metrics": cv_scores
    }
    
    # Save results to JSON file
    saved_file = save_results_to_json(evaluation_results)
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Validation Metrics: {val_scores}")
    print(f"Cross-Validation Metrics: {cv_scores}")
    if saved_file:
        print(f"Detailed results saved to: {saved_file}")
