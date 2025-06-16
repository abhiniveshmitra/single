# app.py
import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Configuration and Initialization ---
# This setup runs only once when the app starts.
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize session state variables if they don't exist.
# This is crucial for preserving state across reruns.
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Rendering ---
# These components are drawn on every rerun, ensuring the UI is always up-to-date.
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
ui_components.render_chat_messages()

# --- Chat Input and Logic ---
# The main application logic is now correctly structured to handle the chat input.
if prompt := st.chat_input("Ask a question, upload a file, or use the mic..."):
    
    # 1. Handle New Chat Creation
    # If no chat is active, create a new one in the database.
    # This happens *without* a rerun, preserving the user's first message.
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = [] # Start with a fresh message list

    # 2. Add and Display User's Message
    # Append the user's message to the session state and display it immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save the user's message to the database for persistence.
    database.add_message(st.session_state.current_chat_id, "user", prompt)

    # 3. Call the AI Agent and Stream the Response
    # Display a spinner while waiting for the Azure service to respond.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The agent is now reliably called with the full message history.
            stream = azure_services.get_chat_completion(
                st.session_state.messages,
                st.session_state.get("vector_store") # Pass vector store if it exists
            )
            # Stream the response to the UI for a ChatGPT-like effect.
            response_content = st.write_stream(stream)
    
    # 4. Save and Handle the Assistant's Response
    # Append the full response to the session state for context in future turns.
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)

    # 5. Handle Text-to-Speech (TTS) if enabled
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                st.audio(audio_data, format="audio/wav")
    
    # Rerun at the very end to reset the input box and finalize the UI state.
    st.rerun()

# --- Microphone Button Logic ---
# This button is separate from the chat input to handle its action cleanly.
if st.button("🎤", key="mic_button", help="Transcribe audio from your microphone"):
    transcribed_text = azure_services.transcribe_audio_from_mic()
    if "Error" not in transcribed_text and transcribed_text:
        # Display the transcribed text as an info message. The user can then
        # easily copy it or re-type it into the chat box. This is the most
        # robust way to handle this with st.chat_input.
        st.info(f"🎤 Transcription: {transcribed_text}")
        st.warning("Please copy the text into the chat box to send.")
    else:
        st.error("Failed to transcribe. Please check microphone permissions and try again.")
