# ingest.py

import os
import sys
import argparse
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

CHROMA_PATH = "chroma_db"

def main():
    # Load environment variables from .env file
    load_dotenv()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Ingest a document into a local ChromaDB vector store.")
    parser.add_argument("file_path", type=str, help="The path to the document file to be ingested.")
    args = parser.parse_args()

    # --- Collection Name ---
    # Create a clean collection name from the filename
    collection_name = os.path.basename(args.file_path).lower()
    collection_name = "".join(c for c in collection_name if c.isalnum() or c in "._-").rstrip()

    print(f"--- Starting ingestion for document: {args.file_path} ---")
    print(f"Target ChromaDB collection: {collection_name}")
    sys.stdout.flush() # Ensure the output is sent immediately

    # --- Step 1: Document Intelligence ---
    print("\nStep 1/4: Analyzing document with Azure Document Intelligence...")
    sys.stdout.flush()
    doc_intel_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intel_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
    try:
        doc_intel_client = DocumentIntelligenceClient(endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key))
        with open(args.file_path, "rb") as f:
            poller = doc_intel_client.begin_analyze_document("prebuilt-layout", f.read(), content_type="application/octet-stream")
        result = poller.result()
        print("[SUCCESS] Analysis complete.")
    except Exception as e:
        print(f"[ERROR] Document analysis failed. Check credentials and file format. Details: {e}")
        return
    sys.stdout.flush()

    # --- Step 2: Content Extraction ---
    print("\nStep 2/4: Extracting text content...")
    sys.stdout.flush()
    full_content = "\n".join([para.content for para in result.paragraphs if para.content])
    if not full_content:
        print("[WARNING] No text content could be extracted from the document.")
        return
    print("[SUCCESS] Content extracted.")
    sys.stdout.flush()

    # --- Step 3: Text Chunking ---
    print("\nStep 3/4: Splitting text into manageable chunks...")
    sys.stdout.flush()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=full_content)
    print(f"[SUCCESS] Content split into {len(chunks)} chunks.")
    sys.stdout.flush()

    # --- Step 4: Embedding and Indexing ---
    print("\nStep 4/4: Creating embeddings with Azure OpenAI and indexing locally...")
    sys.stdout.flush()
    try:
        # --- THE DEFINITIVE FIX ---
        # Be fully explicit when creating the embeddings client to avoid any ambiguity
        # with environment variables or API versions. This is the most robust way.
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-01" # A known stable API version
        )

        Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings_model, 
            collection_name=collection_name, 
            persist_directory=CHROMA_PATH
        )
        print("\n[SUCCESS] Document successfully ingested!")
        print(f"Collection '{collection_name}' is now ready to be loaded in the app.")
    except Exception as e:
        print(f"[ERROR] Failed to create vector store. Check your Azure OpenAI credentials and network. Details: {e}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
# src/azure_services.py

import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

CHROMA_PATH = "chroma_db"

# --- Client Initialization ---
@st.cache_resource
def get_azure_openai_client():
    """Initializes and returns a cached AzureOpenAI client for chat."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

@st.cache_resource
def get_speech_config():
    """Initializes and returns a cached Azure Speech service configuration."""
    return speechsdk.SpeechConfig(
        subscription=os.getenv("SPEECH_KEY"),
        region=os.getenv("SPEECH_REGION")
    )

# --- Function to Load a PRE-COMPUTED ChromaDB collection ---
def load_chroma_collection(collection_name: str):
    """Loads an existing ChromaDB collection that was created by ingest.py."""
    try:
        # --- THE DEFINITIVE FIX ---
        # We must use the exact same explicit instantiation logic here to ensure
        # the app can correctly communicate with the vector database.
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-01"
        )
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings_model,
            collection_name=collection_name
        )
        return db
    except Exception as e:
        st.error(f"Failed to load collection '{collection_name}'. Make sure you have ingested it first. Error: {e}")
        return None

# --- Chat Completion Function ---
def get_chat_completion(messages_from_ui, vector_store: Chroma, image_data=None):
    """Generates a chat response, using a local ChromaDB vector store for context."""
    client = get_azure_openai_client()
    context = ""
    if vector_store:
        last_user_message = messages_from_ui[-1]['content']
        # LangChain uses the embedding_function attached to the vector_store to handle the query
        results = vector_store.similarity_search(last_user_message, k=4)
        if results:
            context = "\n\n".join([doc.page_content for doc in results])

    if context:
        system_prompt = (
            "You are an expert AI assistant for document analysis. Answer the user's questions based ONLY on the provided context. "
            "If the answer is not in the context, state that the document does not contain the answer."
        )
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENT ---\n{context}\n--- END OF CONTEXT ---"
    else:
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question to the best of your ability. "
            "If asked about a document, say that no document has been processed or no relevant information was found."
        )
    
    api_messages = [{"role": "system", "content": system_prompt}] + \
                   [{"role": msg["role"], "content": msg["content"]} for msg in messages_from_ui]
    
    try:
        return client.chat.completions.create(
            model=os.getenv("GPT4_DEPLOYMENT_NAME"),
            messages=api_messages,
            stream=True,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Error connecting to Azure OpenAI: {e}")
        return iter([])

# --- Speech Services ---
def synthesize_text_to_speech(text):
    """Generates speech from text using Azure Speech Services."""
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None

