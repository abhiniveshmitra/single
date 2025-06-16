# app.py
import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Configuration and Initialization ---
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize session state variables
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Rendering ---
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
ui_components.render_chat_messages()

# --- Chat Input and Logic ---
if prompt := st.chat_input("Ask a question, upload a file, or use the mic..."):
    
    # Handle New Chat Creation
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = []

    # Add and Display User's Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    database.add_message(st.session_state.current_chat_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the AI Agent and Stream the Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = azure_services.get_chat_completion(
                st.session_state.messages,
                st.session_state.get("vector_store")
            )
            response_content = st.write_stream(stream)
    
    # --- FIX: Store and Handle Assistant's Response with Audio ---
    assistant_message = {"role": "assistant", "content": response_content}
    
    # Handle Text-to-Speech (TTS) if enabled
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                # Store the audio data directly within the message dictionary
                assistant_message["audio"] = audio_data
    
    # Append the complete message (with or without audio) to state and DB
    st.session_state.messages.append(assistant_message)
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)
    
    # Rerun to finalize the UI
    st.rerun()

# --- Microphone Button Logic (No changes needed here) ---
if st.button("🎤", key="mic_button", help="Transcribe audio from your microphone"):
    transcribed_text = azure_services.transcribe_audio_from_mic()
    if "Error" not in transcribed_text and transcribed_text:
        st.info(f"🎤 Transcription: {transcribed_text}")
        st.warning("Please copy the text into the chat box to send.")
    else:
        st.error("Failed to transcribe. Please check microphone permissions and try again.")
# src/ui_components.py
import streamlit as st
from src import database, document_processor, azure_services

def render_sidebar():
    # No changes are needed in the sidebar logic.
    # This function remains exactly the same as before.
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
                st.rerun()
        
        st.divider()
        st.subheader("Document Q&A (RAG)")
        uploaded_docs = st.file_uploader(
            "Upload PDF or TXT", type=['pdf', 'txt'], accept_multiple_files=True
        )
        if uploaded_docs and not st.session_state.get('vector_store'):
            raw_text = document_processor.get_text_from_files(uploaded_docs)
            if raw_text:
                text_chunks = document_processor.get_text_chunks(raw_text)
                st.session_state.vector_store = document_processor.create_vector_store(text_chunks)
                st.success("Documents ready for Q&A!")

        st.divider()
        st.subheader("Image Analysis")
        uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_image:
            st.image(uploaded_image)
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analysis = azure_services.analyze_image(uploaded_image)
                    if "error" in analysis:
                        st.error(f"Error: {analysis['error']}")
                    else:
                        st.success(f"**Description:** {analysis['description']}")
                        st.write("**Tags:** " + ", ".join(analysis['tags']))

        st.divider()
        st.subheader("Audio Tools")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=False)
        uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
        if uploaded_audio:
            transcribed_text = azure_services.transcribe_audio_file(uploaded_audio)
            st.session_state.prompt_from_audio = transcribed_text

def render_chat_messages():
    """Renders the chat messages from session state, now including audio."""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # --- FIX: Check if the message has audio data and render it ---
            # This makes the audio player persistent across reruns.
            if message["role"] == "assistant" and "audio" in message and message["audio"]:
                st.audio(message["audio"], format="audio/wav")

