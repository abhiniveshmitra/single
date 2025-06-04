import os
import json
import numpy as np
import pandas as pd
from openai import AzureOpenAI
import faiss

# ---- 1. CONFIGURATION ----
dataset_path = r"C:\Users\abhin\OneDrive\Desktop\event-portal\denniswang07\datasets-for-rag\versions\1\output.csv"
faiss_index_path = "rag_index_openai.faiss"
metadata_path = "rag_metadata_openai.json"
column_for_context = "questions"
num_retrievals = 5

# ---- 2. ENVIRONMENT ----
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-10-21"
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
if not api_key or not azure_endpoint:
    raise RuntimeError("Missing Azure OpenAI credentials in environment variables.")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# ---- 3. LOAD DATA ----
print("Loading data and index...")
df = pd.read_csv(dataset_path)
index = faiss.read_index(faiss_index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)
print(f"Loaded {len(df)} data rows, {index.ntotal} vectors in index.")

# ---- 4. START CHAT LOOP ----
conversation = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    query = input("Q: ").strip()
    if not query:
        print("Exiting (empty input).")
        break

    try:
        # Get query embedding
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=[query]
        )
        query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)

        # Retrieve top-K
        D, I = index.search(query_emb.reshape(1, -1), num_retrievals)
        relevant_chunks = []
        for idx in I[0]:
            if 0 <= idx < len(metadata):
                row_idx = metadata[idx].get("row_index", None)
                if row_idx is not None and 0 <= row_idx < len(df):
                    chunk = str(df.iloc[row_idx][column_for_context])
                    relevant_chunks.append(chunk)

        if not relevant_chunks:
            print("No context found for this query.")
            continue

        # Build context and prompt
        context = "\n".join(relevant_chunks)
        prompt = f"Answer this question based on the following context:\n\n{context}\n\nQuestion: {query}"

        conversation.append({"role": "user", "content": prompt})

        # Chat completion
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with your Azure deployment name if different
            messages=conversation
        )
        answer = response.choices[0].message.content
        print("\nAssistant:", answer, "\n")
        conversation.append({"role": "assistant", "content": answer})

        # Optionally trim chat history if needed for context window
        # if len(conversation) > 20:
        #     conversation = [conversation[0]] + conversation[-18:]

    except Exception as e:
        print("Error:", e)
