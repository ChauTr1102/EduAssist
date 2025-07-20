import os
import torch
from datasets import load_dataset

# Load the Vietnamese summarization dataset
print("Loading Vietnamese summarization dataset...")
dataset = load_dataset("OpenHust/vietnamese-summarization")
print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
print(f"Train samples: {len(dataset['train'])}")

# Check the column names in the dataset
train_data = dataset['train']
print(f"Available columns: {train_data.column_names}")
print(f"First example in train: {train_data[0]}")

# Split the dataset
print("Splitting the dataset...")
split_datasets = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']

print(f"After split - Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print(f"Column names in validation dataset: {val_dataset.column_names}")
print(f"First example in validation: {val_dataset[0]}")
