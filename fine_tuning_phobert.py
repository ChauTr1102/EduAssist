import os
import torch
import numpy as np
from datasets import load_dataset
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EncoderDecoderCache
)

# Load the Vietnamese summarization dataset
print("Loading Vietnamese summarization dataset...")
dataset = load_dataset("OpenHust/vietnamese-summarization")


# Split the training data into train and validation sets
print("Splitting training data into train and validation sets...")
train_data = dataset['train']

# Check the column names in the dataset
# print(f"Available columns: {train_data.column_names}")
# print(f"First example keys: {list(train_data[0].keys())}")
# print(f"Sample data structure: {train_data[0]}")

# Instead of using train_test_split, we'll use the built-in Dataset.train_test_split method
# which is designed to work with Hugging Face datasets
print("Using Dataset.train_test_split for proper splitting...")
split_datasets = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Rename the splits for clarity
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']  # The 'test' split here is actually our validation set

# Update the dataset dictionary
dataset['train'] = train_dataset
dataset['validation'] = val_dataset

print(f"After split - Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['validation'])}")
# # Check validation dataset structure
# print("Validation dataset structure:")
# print(f"Column names in validation dataset: {dataset['validation'].column_names}")
# print(f"First example in validation: {dataset['validation'][0]}")

# Use mT5 tokenizer and model for consistency
model_name = "google/mt5-small"
print(f"Loading tokenizer and model: {model_name}")
# Use use_fast=False to avoid issues with the fast tokenizer implementation
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# For sequence-to-sequence tasks, we need to initialize a seq2seq model
# Using mT5 which supports Vietnamese
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set max length for input and output
max_input_length = 512
max_target_length = 128

# Preprocess function to tokenize the data
def preprocess_function(examples):
    # Use the correct column names based on what's available
    if "Document" in examples:
        inputs = examples["Document"]
        targets = examples["Summary"]
    elif "document" in examples:
        inputs = examples["document"]
        targets = examples["summary"]
    else:
        # If columns don't match expected names, try to infer
        keys = list(examples.keys())
        # Assume the first key that isn't Unnamed is the document
        # and the second is the summary
        text_columns = [k for k in keys if not k.startswith("Unnamed")]
        if len(text_columns) >= 2:
            inputs = examples[text_columns[0]]
            targets = examples[text_columns[1]]
        else:
            raise ValueError(f"Could not determine document and summary columns from {keys}")
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Tokenize targets (labels)
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        padding="max_length", 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the dataset
print("Preprocessing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print("Dataset preprocessing completed.")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# Custom optimizer function to use Adam
def get_optimizer(model, learning_rate=5e-5):
    return Adam(model.parameters(), lr=learning_rate)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,  # Updated
    data_collator=data_collator,
    optimizers=(get_optimizer(model, training_args.learning_rate), None)  # (optimizer, lr_scheduler)
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed!")

# Save the fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained("./fine-tuned-vietnamese-summarization")
tokenizer.save_pretrained("./fine-tuned-vietnamese-summarization")
print("Model saved successfully!")

# Evaluate the model
results = trainer.evaluate()
print(results)

# Example of using the model for inference
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True, padding=True)
    
    with torch.no_grad():
        # Convert past_key_values to EncoderDecoderCache
        cache = EncoderDecoderCache.from_legacy_cache(past_key_values)
        summary_ids = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            num_beams=4, 
            min_length=30, 
            max_length=max_target_length, 
            early_stopping=True,
            past_key_values=cache
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Test with a sample text
try:
    # Use validation split for inference
    # Get the first sample from validation dataset
    first_example = dataset["validation"][0]
    
    # Print the keys to see what's available
    print(f"Keys in validation sample: {list(first_example.keys())}")
    
    # Now use the correct column names based on the actual keys
    if "Document" in first_example:
        sample_text = first_example["Document"]
        reference_summary = first_example["Summary"]
    elif "document" in first_example:
        sample_text = first_example["document"]
        reference_summary = first_example["summary"]
    elif "Unnamed: 0" in first_example:
        # Try to find the right columns
        print("Column structure is different than expected.")
        print(f"Available columns: {first_example.keys()}")
        # Default to the first text column if we can't find the right one
        for key in first_example.keys():
            if isinstance(first_example[key], str) and len(first_example[key]) > 100:
                sample_text = first_example[key]
                print(f"Using column '{key}' as the document text.")
                break
        else:
            raise ValueError("Could not find suitable text column in the dataset")
        
        # Use a default reference summary
        reference_summary = "Reference summary not available"
    else:
        # If we can't determine the correct column names, print the structure
        print(f"Dataset structure: {first_example}")
        raise ValueError("Could not determine correct column names")
        
    print("Using validation split for inference...")
    
    summary = generate_summary(sample_text)
    # Comment out or remove print statements that display raw data during inference
    # print(f"Original text: {sample_text[:200]}...")  # Show just the beginning
    # print(f"Generated summary: {summary}")
    # print(f"Reference summary: {reference_summary}")
except Exception as e:
    print(f"Error during inference: {e}")
    print("This might be due to model not being fully trained or compatibility issues.")

# Define past_key_values as None or initialize it properly
past_key_values = None