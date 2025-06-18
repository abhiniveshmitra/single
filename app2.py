# src/ui_components.py

import streamlit as st
import pypdf2
from src import database

def render_sidebar():
    """Renders all sidebar components with the corrected logic."""
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
                st.session_state.document_processor = None
                st.session_state.raw_text = None
                st.session_state.staged_image = None
                st.rerun()

        st.divider()
        st.header("Document Q&A")
        st.write("Upload a document to enable Q&A for this session.")
        
        uploaded_doc = st.file_uploader(
            "Upload a document", 
            type=["pdf"]
        )
        
        if uploaded_doc:
            with st.spinner("Reading document..."):
                try:
                    pdf_reader = pypdf2.PdfReader(uploaded_doc)
                    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                    st.session_state.raw_text = text
                    st.session_state.document_processor = None 
                    st.success(f"'{uploaded_doc.name}' read successfully.")
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")
        
        # THE DEFINITIVE FIX: This safer logic prevents the AttributeError.
        if st.session_state.get('document_processor'):
            st.success("✅ Document is processed and ready for Q&A.")
        elif st.session_state.get('raw_text'):
             st.info("Document loaded. Processing will start with your first question.")

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
    """Renders the complete chat history."""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if "image" in message and message.get("image"):
                st.image(message["image"])
            st.markdown(message["content"])
            if "audio" in message and message.get("audio"):
                st.audio(message["audio"], format="audio/wav")
# app.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components
from src.document_processor import DocumentProcessor

st.set_page_config(page_title="Simple AI Chat", layout="centered")
load_dotenv()

# --- Robust Session State Initialization ---
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

database.init_db()
ui_components.render_sidebar()
st.title("Simple Document Q&A")
ui_components.render_chat_messages()

# --- Main Logic: Process Document if raw_text exists ---
if st.session_state.get("raw_text") and st.session_state.get("document_processor") is None:
    with st.spinner("Processing document: chunking text and creating vector index..."):
        try:
            embedding_model = azure_services.get_embedding_model()
            processor = DocumentProcessor(embedding_model)
            processor.chunk_and_vectorize(st.session_state.raw_text)
            st.session_state.document_processor = processor
            st.session_state.raw_text = None 
            st.rerun()
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            st.stop()

# --- Chat Input and Response Logic ---
if prompt := st.chat_input("Ask a question..."):
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = []

    user_message = {"role": "user", "content": prompt}
    if st.session_state.staged_image:
        user_message["image"] = st.session_state.staged_image
        st.session_state.staged_image = None
        
    st.session_state.messages.append(user_message)
    database.add_message(st.session_state.current_chat_id, "user", prompt)
    
    with st.chat_message("user"):
        if "image" in user_message and user_message.get("image"):
            st.image(user_message["image"])
        st.markdown(prompt)

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

    assistant_message = {"role": "assistant", "content": response_content}
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                assistant_message["audio"] = audio_data
    
    st.session_state.messages.append(assistant_message)
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)
    
    st.rerun()


