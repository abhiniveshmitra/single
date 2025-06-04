import os
import ast
import numpy as np
import pandas as pd
import faiss
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

def load_dataset(csv_path):
    """
    Load the dataset and extract questions.
    """
    df = pd.read_csv(csv_path)
    def extract_questions(row):
        try:
            questions_list = ast.literal_eval(row['questions'])
            return " ".join(questions_list)
        except Exception as e:
            return ""
    df['text'] = df.apply(extract_questions, axis=1)
    return df

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split text into chunks with overlap.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(text_chunks, model):
    """
    Create embeddings and build FAISS index.
    """
    print("Generating embeddings...")
    embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def main():
    # Azure OpenAI Client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Load dataset
    dataset_dir = os.path.join(os.getcwd(), "denniswang", "datasets-for-rag", "versions", "1")
    csv_path = os.path.join(dataset_dir, "output.csv")
    df = load_dataset(csv_path)

    # Chunk data
    all_chunks = []
    for idx, row in df.iterrows():
        all_chunks.extend(chunk_text(row['text']))

    print(f"Total text chunks: {len(all_chunks)}")

    # Build embeddings and index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = build_faiss_index(all_chunks, model)

    # Initialize conversation
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions only using the provided context. "
                "If the answer is not found in the context, respond with: 'I don't know based on the given data.'"
            )
        }
    ]

    while True:
        user_input = input("\nQ: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Embed the user query
        query_embedding = model.encode([user_input], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=5)
        context = "\n\n".join([all_chunks[i] for i in I[0]])

        # Add user message with context
        conversation.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        })

        # Call Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",  # or your deployment name
            messages=conversation,
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": answer})
        print(f"\n🤖 Answer: {answer}\n")

if __name__ == "__main__":
    main()
