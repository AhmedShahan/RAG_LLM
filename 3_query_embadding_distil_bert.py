from transformers import DistilBertTokenizer, DistilBertModel

# DistilBERT টোকেনাইজার এবং মডেল লোড করা
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# উদাহরণস্বরূপ একটি query
query = "Who is Shahan Ahmed"

# টোকেনাইজড ইনপুট তৈরি
inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
# BERT মডেল থেকে আউটপুট নেওয়া
outputs = model(**inputs)

# লাস্ট হিডেন স্টেট থেকে Query embedding
query_embedding = outputs.last_hidden_state.mean(dim=1)
print("Distil Bert Model: ", query_embedding.shape)