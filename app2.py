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

    # --- Step 4: Embedding and Indexing with DETAILED LOGGING ---
    print("\nStep 4/4: Creating embeddings with Azure OpenAI and indexing locally...")
    sys.stdout.flush()
    try:
        # LOGGING: Announce the start of model initialization
        print("Initializing AzureOpenAIEmbeddings model...")
        sys.stdout.flush()
        
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-01"
        )
        
        # LOGGING: Confirm model has been initialized
        print("AzureOpenAIEmbeddings model initialized successfully.")
        sys.stdout.flush()

        # LOGGING: Announce the start of the embedding process
        print("Calling Chroma.from_texts to create embeddings and index chunks...")
        sys.stdout.flush()
        
        Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings_model, 
            collection_name=collection_name, 
            persist_directory=CHROMA_PATH
        )

        # LOGGING: Confirm the process is complete
        print("Chroma.from_texts finished successfully.")
        print("\n[SUCCESS] Document successfully ingested!")
        print(f"Collection '{collection_name}' is now ready to be loaded in the app.")
    except Exception as e:
        # LOGGING: Provide a detailed error message if anything fails
        print(f"[ERROR] An exception occurred during embedding or indexing. Check Azure OpenAI credentials and network. Details: {e}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
