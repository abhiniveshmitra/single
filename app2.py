# app.py

import os
# This MUST be the first line to prevent low-level crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components
from src.document_processor import DocumentProcessor

# --- Page Config and Initialization ---
st.set_page_config(page_title="Simple AI Chat", layout="centered")
load_dotenv()

# --- THE DEFINITIVE FIX: Robust Session State Initialization ---
# This block runs at the top of every rerun to ensure these keys always exist.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "document_processor" not in st.session_state:
    st.session_state.document_processor = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
if "staged_image" not in st.session_state:
    st.session_state.staged_image = None
# --- End of Initialization Block ---

# Initialize the database
database.init_db()

# --- UI Rendering ---
# The UI components are drawn here.
ui_components.render_sidebar()
st.title("Simple Document Q&A")
ui_components.render_chat_messages()

# --- Main Logic: Process Document if Needed ---
# This logic now safely checks for the document processor's existence.
if st.session_state.raw_text and st.session_state.document_processor is None:
    with st.spinner("Processing document: chunking text and creating vector index..."):
        try:
            embedding_model = azure_services.get_embedding_model()
            processor = DocumentProcessor(embedding_model)
            processor.chunk_and_vectorize(st.session_state.raw_text)
            st.session_state.document_processor = processor
            # Rerun to update the sidebar status and clear the raw text
            st.session_state.raw_text = None 
            st.rerun()
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            st.stop()

# --- Chat Input Logic ---
if prompt := st.chat_input("Ask a question..."):
    # If this is the very first message, create a new chat session
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = []

    # Append user message (with image if staged) and save to DB
    user_message = {"role": "user", "content": prompt}
    if st.session_state.staged_image:
        user_message["image"] = st.session_state.staged_image
        st.session_state.staged_image = None
        
    st.session_state.messages.append(user_message)
    database.add_message(st.session_state.current_chat_id, "user", prompt)
    
    with st.chat_message("user"):
        if "image" in user_message:
            st.image(user_message["image"])
        st.markdown(prompt)

    # --- Generate Assistant Response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_chunks = []
            if st.session_state.document_processor:
                context_chunks = st.session_state.document_processor.search(prompt)
            
            stream = azure_services.get_chat_completion(
                st.session_state.messages, 
                context_chunks
            )
            response_content = st.write_stream(stream)

    # Prepare and save assistant message
    assistant_message = {"role": "assistant", "content": response_content}
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                assistant_message["audio"] = audio_data
    
    st.session_state.messages.append(assistant_message)
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)
    
    # Rerun to update the UI state after the response is complete
    st.rerun()
