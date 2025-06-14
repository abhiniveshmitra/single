import streamlit as st
import os
import uuid
import tempfile
from services.openai_chat import chat_completion
from services.file_processing import extract_text
from db.db import init_db, add_chat, add_message, get_chat_history, list_chats, add_file
from dotenv import load_dotenv
import datetime

# For audio input with streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import io
import wave

load_dotenv()
init_db()

st.set_page_config(page_title="Multimodal Enterprise Q&A Copilot", layout="wide")
st.title("🧑‍💼 Multimodal Enterprise File Q&A Copilot")

# --- Sidebar: select or create chat session
with st.sidebar:
    st.header("💬 Chats")
    chats = list_chats()
    chat_titles = [c["title"] for c in chats]
    if "active_chat" not in st.session_state or st.session_state.active_chat not in chat_titles:
        st.session_state.active_chat = chat_titles[0] if chat_titles else None

    chat_idx = st.selectbox(
        "Select chat", 
        options=chat_titles, 
        index=0 if not chats else chat_titles.index(st.session_state.active_chat) if st.session_state.active_chat else 0
    ) if chat_titles else None
    st.session_state.active_chat = chat_idx

    if st.button("➕ New Chat"):
        chat_id = str(uuid.uuid4())[:8]
        title = f"Chat-{chat_id}"
        add_chat(chat_id, title)
        st.session_state.active_chat = title
        st.rerun()

chat_title = st.session_state.active_chat
chat = next((c for c in chats if c["title"] == chat_title), None)
chat_id = chat["id"] if chat else None

if not chat_id:
    st.warning("Please select or create a chat.")
    st.stop()

chat_history = get_chat_history(chat_id)

st.subheader(chat_title)
st.write("Upload PDF, TXT, or type/speak your question.")

# --- File uploader (PDF, TXT, Images)
uploaded_file = st.file_uploader(
    "Upload document (PDF/TXT/JPG/PNG)", 
    type=["pdf", "txt", "jpg", "jpeg", "png"], 
    key="fileup"
)
if uploaded_file:
    text = extract_text(uploaded_file, uploaded_file.name)
    if not text:
        st.error("Could not extract text from file.")
    else:
        # Save file to disk
        storage_dir = os.path.join("storage", chat_id)
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_file(chat_id, uploaded_file.name, file_path, uploaded_file.type)
        add_message(chat_id, "system", f"[File uploaded: {uploaded_file.name}]\n{text[:1000]}...")
        st.success(f"File '{uploaded_file.name}' processed and saved.")

# --- Display chat history
for msg in chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Copilot:** {msg['content']}")
    else:
        st.markdown(f"`{msg['content']}`")

# --- Voice input (mic) using streamlit-webrtc (NO client_settings argument)
st.write("Type your question or use your microphone below:")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

    def get_wav_bytes(self):
        if not self.recorded_frames:
            return None
        pcm_audio = np.concatenate(self.recorded_frames)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(16000)
            wf.writeframes(pcm_audio.tobytes())
        return wav_buf.getvalue()

audio_ctx = webrtc_streamer(
    key="speech-to-text",
    mode="SENDONLY",
    audio_receiver_size=256,
    async_processing=False,
    audio_processor_factory=AudioProcessor,
)

# Store/handle audio for transcription
if audio_ctx and audio_ctx.audio_processor:
    if st.button("Transcribe Last Recording"):
        wav_bytes = audio_ctx.audio_processor.get_wav_bytes()
        if wav_bytes:
            from services.speech import speech_to_text
            transcript = speech_to_text(wav_bytes)
            add_message(chat_id, "user", transcript)
            st.success(f"Transcribed: {transcript}")
            st.rerun()
        else:
            st.info("No audio captured yet. Press the mic, speak, then press 'Transcribe Last Recording'.")

# --- Send user message (text box)
user_input = st.text_input("Your question:", key="chatbox")
if st.button("Send") and user_input.strip():
    add_message(chat_id, "user", user_input)
    # Get full chat history (system/user/assistant messages only)
    messages = [{"role": m["role"], "content": m["content"]} for m in get_chat_history(chat_id)]
    response = chat_completion(messages)
    add_message(chat_id, "assistant", response)
    st.rerun()
