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
        print("✅ Analysis complete.")
    except Exception as e:
        print(f"❌ ERROR: Document analysis failed. Check credentials and file format. Details: {e}")
        return
    sys.stdout.flush()

    # --- Step 2: Content Extraction ---
    print("\nStep 2/4: Extracting text content...")
    sys.stdout.flush()
    full_content = "\n".join([para.content for para in result.paragraphs if para.content])
    if not full_content:
        print("⚠️ WARNING: No text content could be extracted.")
        return
    print("✅ Content extracted.")
    sys.stdout.flush()

    # --- Step 3: Text Chunking ---
    print("\nStep 3/4: Splitting text into manageable chunks...")
    sys.stdout.flush()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=full_content)
    print(f"✅ Content split into {len(chunks)} chunks.")
    sys.stdout.flush()

    # --- Step 4: Embedding and Indexing ---
    print("\nStep 4/4: Creating embeddings with Azure OpenAI and indexing locally...")
    sys.stdout.flush()
    try:
        embeddings_model = AzureOpenAIEmbeddings(azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"), openai_api_version="2024-05-01-preview")
        Chroma.from_texts(texts=chunks, embedding=embeddings_model, collection_name=collection_name, persist_directory=CHROMA_PATH)
        print("\n🎉 SUCCESS! 🎉")
        print(f"Document successfully ingested into collection: '{collection_name}'")
        print("You can now load this collection in the Streamlit app.")
    except Exception as e:
        print(f"❌ ERROR: Failed to create vector store. Check Azure OpenAI credentials and network. Details: {e}")
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
    Renders all components in the sidebar, including chat history,
    the new automated document ingestion workflow, and other tools.
    """
    with st.sidebar:
        st.header("Azure AI Assistant")
        
        # Button to start a new chat session, clearing all state
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.subheader("Previous Chats")
        # Load and display previous chat sessions from the database
        chats = database.get_chats()
        for chat_id, title, _ in chats:
            if st.button(title, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = database.get_messages(chat_id)
                # Clear any session-specific data like the vector store
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
                # --- This is the new, automated workflow ---
                with st.spinner(f"Processing {uploaded_doc.name}... This may take a while."):
                    # 1. Save the uploaded file to a temporary location so our script can access it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_doc.getvalue())
                        tmp_file_path = tmp_file.name

                    # 2. Run ingest.py as a background process and show live output
                    with st.expander("Show Processing Details", expanded=True):
                        # Construct the command, ensuring it uses the SAME Python executable as Streamlit
                        # This avoids the "two pythons" problem
                        command = [sys.executable, "ingest.py", tmp_file_path]
                        
                        # Use Popen to run the script in the background and capture its output
                        process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding='utf-8'
                        )
                        
                        # Stream the output from the script to the Streamlit UI in real-time
                        output_placeholder = st.empty()
                        full_output = ""
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                            if line:
                                full_output += line
                                output_placeholder.code(full_output)
                        
                        process.wait() # Wait for the process to finish
                        
                # 3. Clean up the temporary file
                os.remove(tmp_file_path)

                # 4. Set the collection name for the app to use on success
                if process.returncode == 0:
                    # Recreate the collection name in the same way ingest.py does
                    collection_name = os.path.basename(uploaded_doc.name).lower()
                    collection_name = "".join(c for c in collection_name if c.isalnum() or c in "._-").rstrip()
                    st.session_state.collection_name = collection_name
                    st.success(f"'{uploaded_doc.name}' processed successfully!")
                    st.rerun()
                else:
                    st.error("Processing failed. See details above.")
        
        st.divider()
        st.header("Load Processed Document")
        # UI to manually load a collection if needed (useful fallback)
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

        # The image analysis logic remains the same and is unaffected[1]
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
        
        # Audio tools logic also remains the same
        st.subheader("Audio Tools")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=False)


def render_chat_messages():
    """
    Renders the complete chat history from the session state.
    This function did not require any changes.
    """
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message and message["image"]:
                st.image(message["image"])
            
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "audio" in message and message["audio"]:
                st.audio(message["audio"], format="audio/wav")

