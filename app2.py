# --- DISABLE LANGSMITH/LANGCHAIN CLOUD TRACING (MUST BE FIRST!) ---
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import json
import uuid
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain and related imports
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------------- PART 1: DATA GENERATION --------------------------
def generate_call_records_to_file(num_records=100, output_filename="sample_cdrs_for_analysis.jsonl"):
    sample_upns = [
        "adele.vance@contoso.com", "alex.wilber@contoso.com", "megan.bowen@contoso.com",
        "lynne.robbins@contoso.com", "diego.siciliani@contoso.com", "patti.ferguson@contoso.com"
    ]
    with open(output_filename, 'w') as f:
        for _ in range(num_records):
            call_type = random.choice(["groupCall", "peerToPeer"])
            start_time = datetime.utcnow() - timedelta(minutes=random.randint(5, 120))
            jitter_value = random.uniform(0.005, 0.080) if random.random() > 0.3 else None
            flattened_data = {
                "conferenceId": str(uuid.uuid4()),
                "callType": call_type,
                "startDateTime": start_time.isoformat() + "Z",
                "modalities": random.sample(["audio", "video", "videoBasedScreenSharing"], k=random.randint(1, 3)),
                "organizerUPN": random.choice(sample_upns),
                "clientPlatform": random.choice(["windows", "macOS", "android"]),
                "averageJitter": f"PT{jitter_value:.3f}S" if jitter_value else None,
                "averageAudioDegradation": round(random.uniform(0.1, 1.0), 2) if random.random() > 0.6 else None,
            }
            f.write(json.dumps(flattened_data) + '\n')

# --------------------- PART 2: DATA LOADING AND EXTRACTION --------------------
def load_cdr_documents_from_jsonl(filepath):
    documents = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            page_content = (
                f"Call record for user {record.get('organizerUPN')} on {record.get('clientPlatform')}. "
                f"Call type was {record.get('callType')}. "
                f"Quality metrics show average jitter of {record.get('averageJitter', 'N/A')} "
                f"and audio degradation of {record.get('averageAudioDegradation', 'N/A')}."
            )
            documents.append(Document(page_content=page_content, metadata=record))
    return documents

# ----------------- PART 3: EMBEDDINGS AND FAISS VECTOR STORE ------------------
def build_faiss_index(documents, azure_embedding_args, faiss_index_path):
    embeddings_model = AzureOpenAIEmbeddings(**azure_embedding_args)
    vector_store = FAISS.from_documents(documents, embeddings_model)
    vector_store.save_local(faiss_index_path)
    return vector_store

# ---------------------- PART 4: QUERYING THE RAG CHAIN ------------------------
def query_rag_chain(vector_store, azure_chat_args, question):
    llm = AzureChatOpenAI(**azure_chat_args)
    prompt = hub.pull("rlm/rag-prompt")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    return result["answer"]

# -------------------------------- MAIN BLOCK ----------------------------------
if __name__ == "__main__":
    load_dotenv()

    cdr_filename = "sample_cdrs_for_analysis.jsonl"
    faiss_index_path = "faiss_cdr_index"

    # Azure OpenAI environment config
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

    # 1. Generate sample data
    print("--- 1. Generating sample call records ---")
    generate_call_records_to_file(num_records=100, output_filename=cdr_filename)

    # 2. Load docs and build index
    print("--- 2. Loading documents and building FAISS index ---")
    docs = load_cdr_documents_from_jsonl(cdr_filename)
    vector_store = build_faiss_index(docs, azure_embedding_args, faiss_index_path)

    # 3. General query
    general_q = (
        "Analyze the call data. Are there any users or platforms with notably poor call quality? "
        "Summarize your findings and suggest an action."
    )
    print("\n=======================================")
    print("         GENERAL ACTIONABLE INSIGHT")
    print("=======================================")
    print(query_rag_chain(vector_store, azure_chat_args, general_q))

    # 4. User-specific query
    specific_user = "adele.vance@contoso.com"
    specific_q = (
        f"Analyze all call records specifically for the user {specific_user} "
        "and summarize any quality issues you find in their records."
    )
    print("\n=======================================")
    print(f"    ACTIONABLE INSIGHT for {specific_user}")
    print("=======================================")
    print(query_rag_chain(vector_store, azure_chat_args, specific_q))
    print("=======================================\n")
