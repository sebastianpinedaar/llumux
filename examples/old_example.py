from dataset import load_data
from transformers import BertModel, BertTokenizer
import torch

dataloader = load_data.get_dataloader_from_hf(batch_size=2)
model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch = next(iter(dataloader))

prompt = batch["prompt"]
token = tokenizer(prompt,  return_tensors = 'pt', padding=True)
prediction = model(**token)


#prediction.last_hidden_state.shape = [2, 277, 768]