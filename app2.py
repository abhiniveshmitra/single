#build_embeddings.py
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json

def load_flattened_documents(filepath):
    """
    Loads flattened records from a .jsonl file into LangChain Documents.
    Each line should have a 'summary' field for embedding.
    """
    documents = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            summary = record.get('summary')
            documents.append(Document(page_content=summary, metadata={
                "organizerUPN": record.get("organizerUPN"),
                "conferenceId": record.get("conferenceId")
            }))
    return documents

def build_faiss_index(documents, azure_embedding_args, faiss_index_path):
    embeddings_model = AzureOpenAIEmbeddings(**azure_embedding_args)
    vector_store = FAISS.from_documents(documents, embeddings_model)
    vector_store.save_local(faiss_index_path)
    return vector_store

if __name__ == "__main__":
    load_dotenv()
    flattened_jsonl = "flattened_cdrs.jsonl"
    faiss_index_path = "faiss_cdr_index"
    azure_embedding_args = {
        "azure_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "openai_api_version": os.getenv("OPENAI_API_VERSION"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    }

    docs = load_flattened_documents(flattened_jsonl)
    build_faiss_index(docs, azure_embedding_args, faiss_index_path)
    print(f"FAISS vector index built from {flattened_jsonl} and saved as {faiss_index_path}!")
#query_insight.py
# --- DISABLE LANGSMITH CLOUD ---
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_vector_store(faiss_index_path, azure_embedding_args):
    # Rebuild vector store (no need to recompute embeddings)
    embeddings_model = __import__("langchain_openai").AzureOpenAIEmbeddings(**azure_embedding_args)
    return FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)

def query_rag_chain(vector_store, azure_chat_args, question):
    local_prompt_template = """
You are an expert Teams call analytics assistant.
- Use ONLY the provided call record data to answer.
- Your answer must contain actionable insights.
- If you detect a pattern (e.g., high packet loss during a certain hour, one platform/headset with bad performance), mention it and suggest a concrete action (e.g., 'change your headset', 'avoid 3pm calls', 'try a different device').
- If the data is insufficient, say so politely.

Here are the relevant call records:
{context}

Question: {input}
Actionable Insight:
"""
    prompt = PromptTemplate.from_template(local_prompt_template)
    llm = AzureChatOpenAI(**azure_chat_args)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    return result["answer"]

if __name__ == "__main__":
    load_dotenv()
    faiss_index_path = "faiss_cdr_index"

    azure_embedding_args = {
        "azure_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "openai_api_version": os.getenv("OPENAI_API_VERSION"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    }
    azure_chat_args = {
        "azure_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "openai_api_version": os.getenv("OPENAI_API_VERSION"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "temperature": 0,
    }

    vector_store = load_vector_store(faiss_index_path, azure_embedding_args)

    # General insight
    print("\n=======================================")
    print("         GENERAL ACTIONABLE INSIGHT")
    print("=======================================")
    print(query_rag_chain(
        vector_store, azure_chat_args,
        "Give actionable suggestions based on all call records. Do you see issues with certain platforms, headsets, or time slots?"
    ))

    # Query for a specific user
    specific_user = "adele.vance@contoso.com"
    print("\n=======================================")
    print(f"    ACTIONABLE INSIGHT for {specific_user}")
    print("=======================================")
    print(query_rag_chain(
        vector_store, azure_chat_args,
        f"Analyze all call records for {specific_user}. Summarize any technical problems or patterns and what should be done to improve their experience."
    ))
    print("=======================================\n")

