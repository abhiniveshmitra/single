from dotenv import load_dotenv
load_dotenv()
import os
import json
import pandas as pd
import numpy as np
import faiss
from openai import AzureOpenAI
from tqdm import tqdm

# CONFIGURATION
dataset_path = "mini_rag.csv"
column_to_embed = "questions"
faiss_index_path = "rag_index_openai.faiss"
metadata_path = "rag_metadata_openai.json"

# ENVIRONMENT
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
if not api_key or not azure_endpoint:
    raise RuntimeError("Missing Azure OpenAI credentials in environment variables.")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# LOAD DATA
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
if column_to_embed not in df.columns:
    raise ValueError(f"Column '{column_to_embed}' not found in CSV.")
texts = df[column_to_embed].astype(str).tolist()
print(f"Loaded {len(texts)} chunks.")

# EMBEDDING GENERATION
all_embeddings = []
print("Generating embeddings with OpenAI...")
for text in tqdm(texts, desc="Embedding", unit="chunk"):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Your deployment name here
        input=[text]
    )
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    all_embeddings.append(emb)
embeddings_matrix = np.vstack(all_embeddings)
print("Embeddings shape:", embeddings_matrix.shape)

# BUILD FAISS INDEX
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
index.add(embeddings_matrix)
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")

# SAVE METADATA
metadata = [{"row_index": i} for i in range(len(texts))]
with open(metadata_path, "w") as f:
    json.dump(metadata, f)
print(f"Metadata saved to {metadata_path}")

print("\n✅ Index builder completed successfully!")
