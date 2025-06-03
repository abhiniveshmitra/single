from datasets import load_dataset
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Azure OpenAI client setup (same as before)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# 2. Load the RAG dataset
ds = load_dataset("neural-bridge/rag-dataset-12000")
# For demo, use the training set (could use test set too)
dataset = ds['train']

# 3. Test the pipeline with one sample (or loop through a few)
for idx in range(3):  # Try first 3 examples for demo
    sample = dataset[idx]
    context = sample['context']
    question = sample['question']
    gold_answer = sample['answer']

    print(f"\n--- Example {idx+1} ---")
    print("Context:", context[:300], "...")  # Print only first 300 chars for brevity
    print("Question:", question)
    print("Reference Answer:", gold_answer)

    # 4. Build prompt for RAG
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200
    )
    ai_answer = response.choices[0].message.content
    print("AI Answer:", ai_answer)

    # Optionally compare/give a score (manual for now)
    print("-----")

