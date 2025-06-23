# Step 0: Ensure all necessary packages are installed.
# You can run this command in your terminal or a notebook cell.
# !pip install langchain langchain-openai langchain-community faiss-cpu jsonlines python-dotenv

import os
import json
import uuid
import random
from datetime import datetime, timedelta

# LangChain and related imports
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# --- PART 1: DATA GENERATION ---

def generate_call_records_to_file(num_records=100, output_filename="sample_cdrs_for_analysis.jsonl"):
    """
    Generates realistic, flattened Microsoft Teams call records and saves them
    to a specified output file in JSONL format.
    """
    print(f"--- 1. Generating {num_records} sample call records into '{output_filename}' ---")
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
                "conferenceId": str(uuid.uuid4()), "callType": call_type,
                "startDateTime": start_time.isoformat() + "Z",
                "modalities": random.sample(["audio", "video", "videoBasedScreenSharing"], k=random.randint(1, 3)),
                "organizerUPN": random.choice(sample_upns),
                "clientPlatform": random.choice(["windows", "macOS", "android"]),
                "averageJitter": f"PT{jitter_value:.3f}S" if jitter_value else None,
                "averageAudioDegradation": round(random.uniform(0.1, 1.0), 2) if random.random() > 0.6 else None,
            }
            f.write(json.dumps(flattened_data) + '\n')
    print("--- Data generation complete. ---")


# --- PART 2: DATA LOADING AND EXTRACTION ---

def load_cdr_documents_from_jsonl(filepath):
    """Loads CDR data from a JSONL file into LangChain Documents."""
    print(f"--- 2. Loading and extracting data from '{filepath}' ---")
    documents = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            # This text summary is what the AI searches against. Including the UPN here is key.
            page_content = f"Call record for user {record.get('organizerUPN')} on {record.get('clientPlatform')}. " \
                           f"Call type was {record.get('callType')}. " \
                           f"Quality metrics show average jitter of {record.get('averageJitter', 'N/A')} " \
                           f"and audio degradation of {record.get('averageAudioDegradation', 'N/A')}."
            documents.append(Document(page_content=page_content, metadata=record))
    print(f"--- Successfully loaded {len(documents)} documents. ---")
    return documents


# --- PART 3: MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Load environment variables from your .env file
    load_dotenv()
    
    cdr_filename = "sample_cdrs_for_analysis.jsonl"
    faiss_index_path = "faiss_cdr_index"

    # Step 1: Generate the data file
    generate_call_records_to_file(num_records=100, output_filename=cdr_filename)

    # Step 2: Load documents from the generated file
    documents = load_cdr_documents_from_jsonl(cdr_filename)

    # Step 3: Initialize connections to your Azure AI models
    print("--- 3. Initializing Azure AI models ---")
    embeddings_model = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0
    )

    # Step 4: Create and save the local vector store
    print("--- 4. Creating local vector store ---")
    vector_store = FAISS.from_documents(documents, embeddings_model)
    vector_store.save_local(faiss_index_path)
    print(f"   Index created and saved to '{faiss_index_path}'.")

    # Step 5: Build the main RAG chain
    print("--- 5. Building the RAG chain ---")
    retrieval_qa_chat_prompt = hub.pull("rlm/rag-prompt")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # Step 6: Ask a general question for overall insights
    print("\n--- 6. Querying the AI Agent for general insights ---")
    general_question = "Analyze the call data. Are there any users or platforms with notably poor call quality? Summarize your findings and suggest an action."
    result = retrieval_chain.invoke({"input": general_question})
    
    print("\n=======================================")
    print("         GENERAL ACTIONABLE INSIGHT")
    print("=======================================")
    print(result["answer"])
    
    # --- THIS IS THE KEY PART FOR YOUR REQUEST ---
    # Step 7: Ask a specific question with a user ID in natural language
    print("\n--- 7. Querying the AI Agent for a specific user by ID ---")
    specific_user = "adele.vance@contoso.com"
    specific_question = f"Analyze all call records specifically for the user {specific_user} and summarize any quality issues you find in their records."
    
    specific_result = retrieval_chain.invoke({"input": specific_question})
    
    print("\n=======================================")
    print(f"    ACTIONABLE INSIGHT for {specific_user}")
    print("=======================================")
    print(specific_result["answer"])
    print("=======================================\n")
