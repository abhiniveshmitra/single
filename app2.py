# ingest.py

import os
import sys
import argparse
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

CHROMA_PATH = "chroma_db"

def main():
    # --- THE RADICAL CHANGE: Get all info from command-line arguments, not .env ---
    parser = argparse.ArgumentParser(description="Ingest a document into a local ChromaDB vector store.")
    parser.add_argument("file_path", type=str, help="The path to the document file.")
    parser.add_argument("--doc-intel-endpoint", required=True, type=str, help="Azure Document Intelligence endpoint.")
    parser.add_argument("--doc-intel-key", required=True, type=str, help="Azure Document Intelligence key.")
    parser.add_argument("--openai-endpoint", required=True, type=str, help="Azure OpenAI endpoint.")
    parser.add_argument("--openai-key", required=True, type=str, help="Azure OpenAI API key.")
    parser.add_argument("--openai-deployment", required=True, type=str, help="Azure OpenAI embedding deployment name.")
    args = parser.parse_args()

    # Create a clean collection name from the filename
    collection_name = os.path.basename(args.file_path).lower()
    collection_name = "".join(c for c in collection_name if c.isalnum() or c in "._-").rstrip()

    print(f"--- Starting ingestion for document: {args.file_path} ---")
    print(f"Target ChromaDB collection: {collection_name}")
    sys.stdout.flush()

    # --- Step 1: Document Intelligence ---
    print("\nStep 1/4: Analyzing document...")
    sys.stdout.flush()
    try:
        doc_intel_client = DocumentIntelligenceClient(endpoint=args.doc_intel_endpoint, credential=AzureKeyCredential(args.doc_intel_key))
        with open(args.file_path, "rb") as f:
            poller = doc_intel_client.begin_analyze_document("prebuilt-layout", f.read(), content_type="application/octet-stream")
        result = poller.result()
        print("[SUCCESS] Analysis complete.")
    except Exception as e:
        print(f"[ERROR] Document analysis failed. Check Doc Intel credentials. Details: {e}")
        sys.exit(1) # Exit with an error code
    sys.stdout.flush()

    # --- Step 2: Content Extraction ---
    print("\nStep 2/4: Extracting text content...")
    sys.stdout.flush()
    full_content = "\n".join([para.content for para in result.paragraphs if para.content])
    if not full_content:
        print("[WARNING] No text content could be extracted from the document.")
        sys.exit(1)
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
    print("\nStep 4/4: Creating embeddings and indexing locally...")
    sys.stdout.flush()
    try:
        # Use the arguments passed directly from the parent process
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=args.openai_key,
            azure_deployment=args.openai_deployment,
            azure_endpoint=args.openai_endpoint,
            openai_api_version="2024-02-01"
        )
        Chroma.from_texts(texts=chunks, embedding=embeddings_model, collection_name=collection_name, persist_directory=CHROMA_PATH)
        print("\n[SUCCESS] Document successfully ingested!")
        print(f"Collection '{collection_name}' is now ready to be loaded in the app.")
    except Exception as e:
        print(f"[ERROR] Failed to create vector store. Check your Azure OpenAI deployment name and credentials. Details: {e}")
        sys.exit(1)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
# src/ui_components.py

import streamlit as st
import tempfile
import subprocess
import sys
import os
from src import database, azure_services

def render_sidebar():
    """
    Renders all components in the sidebar, now with a robust, automated
    ingestion workflow that securely passes credentials.
    """
    with st.sidebar:
        st.header("Azure AI Assistant")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.subheader("Previous Chats")
        chats = database.get_chats()
        for chat_id, title, _ in chats:
            if st.button(title, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = database.get_messages(chat_id)
                st.session_state.vector_store = None
                st.session_state.collection_name = None
                st.session_state.staged_image = None
                st.rerun()
        
        st.divider()
        st.header("Automated Document Ingestion")
        st.write("Upload a document to automatically process and index it for Q&A.")

        uploaded_doc = st.file_uploader(
            "Upload PDF, DOCX, or TXT file",
            type=['pdf', 'docx', 'txt'],
            key="automated_doc_uploader"
        )
        
        if uploaded_doc:
            if st.button(f"Process '{uploaded_doc.name}'", use_container_width=True):
                with st.spinner(f"Processing {uploaded_doc.name}... This may take a while."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_doc.getvalue())
                        tmp_file_path = tmp_file.name

                    with st.expander("Show Processing Details", expanded=True):
                        # --- THE DEFINITIVE FIX ---
                        # 1. Read the credentials in the main app, where load_dotenv() works.
                        doc_intel_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
                        doc_intel_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
                        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                        openai_key = os.getenv("AZURE_OPENAI_API_KEY")
                        openai_deployment = os.getenv("ADA_DEPLOYMENT_NAME")

                        # 2. Build the command list, passing credentials as arguments.
                        command = [
                            sys.executable, "ingest.py", tmp_file_path,
                            "--doc-intel-endpoint", doc_intel_endpoint,
                            "--doc-intel-key", doc_intel_key,
                            "--openai-endpoint", openai_endpoint,
                            "--openai-key", openai_key,
                            "--openai-deployment", openai_deployment
                        ]
                        
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding='utf-8',
                            errors='replace' # Add robustness for character encoding issues
                        )
                        
                        output_placeholder = st.empty()
                        full_output = ""
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                            if line:
                                full_output += line
                                output_placeholder.code(full_output)
                        
                        process.wait()
                        
                os.remove(tmp_file_path)

                if process.returncode == 0:
                    collection_name = os.path.basename(uploaded_doc.name).lower()
                    collection_name = "".join(c for c in collection_name if c.isalnum() or c in "._-").rstrip()
                    st.session_state.collection_name = collection_name
                    st.success(f"'{uploaded_doc.name}' processed successfully!")
                    st.rerun()
                else:
                    st.error("Processing failed. See details above.")
        
        st.divider()
        st.header("Load Processed Document")
        collection_name_input = st.text_input(
            "Or Enter Collection Name to Load:",
            help="e.g., 'pandas.pdf'"
        )
        if st.button("Load Document Manually", use_container_width=True):
            if collection_name_input:
                st.session_state.collection_name = collection_name_input
                st.rerun()

        if st.session_state.get("collection_name"):
            st.success(f"Active Document: {st.session_state.collection_name}")

        st.divider()
        st.header("Tools & Settings")

        st.subheader("Image Analysis")
        uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'], key="img_uploader")
        if uploaded_image:
            st.session_state.staged_image = uploaded_image.read()
            st.success("Image staged for the next message.")

        if st.session_state.get("staged_image"):
            st.image(st.session_state.staged_image, caption="This image is staged.")
            if st.button("Clear Staged Image", use_container_width=True):
                st.session_state.staged_image = None
                st.rerun()
        
        st.subheader("Audio Tools")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=False)


def render_chat_messages():
    """ Renders the complete chat history from the session state. """
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message and message["image"]:
                st.image(message["image"])
            
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "audio" in message and message["audio"]:
                st.audio(message["audio"], format="audio/wav")
