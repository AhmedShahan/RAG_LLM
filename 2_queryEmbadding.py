## Install transformer
## install torch
import torch 

from transformers import BertTokenizer, BertModel

# BERT টোকেনাইজার এবং মডেল লোড করা
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# উদাহরণস্বরূপ একটি query
query = "Who is Shahan Ahmed"

# টোকেনাইজড ইনপুট তৈরি
inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
# BERT মডেল থেকে আউটপুট নেওয়া
outputs = model(**inputs)

# লাস্ট হিডেন স্টেট থেকে Query embedding
query_embedding = outputs.last_hidden_state.mean(dim=1)
print(query_embedding)