# app.py

import os
# --- THE RADICAL AND FINAL FIX ---
# This line MUST be at the absolute top of your script, before any other imports
# that might load conflicting libraries (like numpy, torch, etc.).
# It tells the system's math libraries to allow multiple versions to be loaded,
# resolving the underlying DLL conflict that causes the app to crash silently.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Config and Initialization ---
# This sets up the browser tab title and loads your secret credentials.
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize all session state variables at the beginning to prevent errors.
# This ensures that even on the first run, these keys exist.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "staged_image" not in st.session_state:
    st.session_state.staged_image = None

# This logic makes the document context persistent within a session.
# If the app reruns, it will try to reload the last used ChromaDB collection.
if st.session_state.collection_name and not st.session_state.vector_store:
    with st.spinner(f"Reloading '{st.session_state.collection_name}' from local storage..."):
        st.session_state.vector_store = azure_services.load_chroma_collection(st.session_state.collection_name)

# --- UI Rendering ---
# These two lines draw the entire user interface.
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
st.info("To ask questions about a document, please upload and process it using the options in the sidebar.")
ui_components.render_chat_messages()

# --- User Input Handling ---
# This block captures when the user types something and hits Enter.
if prompt := st.chat_input("Ask a question..."):
    
    # If this is the very first message, create a new chat session in the database.
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = []

    # Prepare the user message object, including the staged image if one exists.
    # This reflects your expertise in building AI systems with image analysis capabilities[2].
    user_message = {"role": "user", "content": prompt}
    if st.session_state.staged_image:
        user_message["image"] = st.session_state.staged_image
        st.session_state.staged_image = None
        
    # Add the message to the session state and database, then rerun to display it immediately.
    st.session_state.messages.append(user_message)
    database.add_message(st.session_state.current_chat_id, "user", prompt)
    st.rerun()

# --- AI Response Generation ---
# This block runs only after a user message has been submitted and displayed.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the main AI service, passing the chat history and the active vector store.
            stream = azure_services.get_chat_completion(
                messages_from_ui=st.session_state.messages,
                vector_store=st.session_state.get("vector_store")
            )
            response_content = st.write_stream(stream)
    
    # Prepare the assistant's message object.
    assistant_message = {"role": "assistant", "content": response_content}
    
    # If TTS is enabled, generate the audio and add it to the message object.
    # This leverages your experience with text-to-speech technologies[1].
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                assistant_message["audio"] = audio_data
    
    # Save the complete assistant message to the session state and database.
    st.session_state.messages.append(assistant_message)
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)
    
    # Final rerun to settle the UI and wait for the next user input.
    st.rerun()
