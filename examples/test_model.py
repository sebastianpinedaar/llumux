from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)  