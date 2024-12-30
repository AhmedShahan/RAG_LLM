## install pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# SBERT মডেল লোড করা
model = SentenceTransformer('all-MiniLM-L6-v2')
# উদাহরণ query
query = "Who is Shahan Ahmed"


# Query embedding তৈরি করা
query_embedding = model.encode(query)

# Embedding প্রিন্ট করা
print(query_embedding.shape)
