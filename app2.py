# src/azure_services.py
import os
import base64
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk

# --- Client Initialization ---
@st.cache_resource
def get_azure_openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

@st.cache_resource
def get_speech_config():
    return speechsdk.SpeechConfig(
        subscription=os.getenv("SPEECH_KEY"),
        region=os.getenv("SPEECH_REGION")
    )

# --- OpenAI Service (Now with Vision) ---
def get_chat_completion(messages_from_ui, vector_store=None, image_data=None):
    client = get_azure_openai_client()
    system_prompt = "You are an expert AI assistant. Answer the user's questions based on any text and images provided. Be descriptive and helpful."

    # RAG Context
    if vector_store and messages_from_ui and messages_from_ui[-1]['role'] == 'user':
        user_query = messages_from_ui[-1]['content']
        docs = vector_store.similarity_search(user_query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENTS ---\n{context}\n--- END OF CONTEXT ---"

    # Prepare a clean list of messages for the API, handling the image if present.
    api_messages = []
    
    # Check the last user message for an image.
    last_user_message_index = -1
    for i in reversed(range(len(messages_from_ui))):
        if messages_from_ui[i]['role'] == 'user':
            last_user_message_index = i
            break
    
    for i, msg in enumerate(messages_from_ui):
        # Attach the image only to the last user message.
        if msg['role'] == 'user' and i == last_user_message_index and image_data:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            api_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": msg['content']},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            })
        else:
            # For all other messages, just use the text content.
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    full_api_messages = [{"role": "system", "content": system_prompt}] + api_messages
    
    return client.chat.completions.create(
        model=os.getenv("GPT4_DEPLOYMENT_NAME"),
        messages=full_api_messages,
        stream=True,
        max_tokens=1024,
        temperature=0.7
    )

# --- Speech Services ---
def transcribe_audio_from_mic():
    speech_config = get_speech_config()
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Listening... Speak into your microphone.")
    result = recognizer.recognize_once_async().get()
    st.info("Processing complete.")
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Error: Could not recognize speech."

def synthesize_text_to_speech(text):
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None

def transcribe_audio_file(audio_file):
    speech_config = get_speech_config()
    audio_stream = speechsdk.audio.PushAudioInputStream()
    audio_stream.write(audio_file.read())
    audio_stream.close()
    audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Transcribing audio file...")
    result = recognizer.recognize_once_async().get()
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Could not transcribe audio."
# src/ui_components.py
import streamlit as st
from src import database, document_processor, azure_services

def render_sidebar():
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
                st.session_state.staged_image = None
                st.rerun()
        
        st.divider()
        st.subheader("Add Context to Your Next Message")
        
        # Simplified Image Uploader to "stage" the image for the next prompt.
        uploaded_image = st.file_uploader(
            "Upload an Image", type=['jpg', 'png', 'jpeg'], key="img_uploader"
        )
        if uploaded_image:
            st.session_state.staged_image = uploaded_image.read()
            st.success("Image staged. It will be sent with your next message.")
        
        # Display the staged image and provide a button to clear it.
        if st.session_state.get("staged_image"):
            st.image(st.session_state.staged_image, caption="This image is staged.")
            if st.button("Clear Staged Image", use_container_width=True):
                st.session_state.staged_image = None
                st.rerun()
        
        st.divider()
        st.subheader("Document Q&A (RAG)")
        uploaded_docs = st.file_uploader(
            "Upload PDF or TXT", type=['pdf', 'txt'], accept_multiple_files=True, key="doc_uploader"
        )
        if uploaded_docs and not st.session_state.get('vector_store'):
            raw_text = document_processor.get_text_from_files(uploaded_docs)
            if raw_text:
                text_chunks = document_processor.get_text_chunks(raw_text)
                st.session_state.vector_store = document_processor.create_vector_store(text_chunks)
                st.success("Documents ready for Q&A!")

        st.divider()
        st.subheader("Audio Tools")
        st.session_state.tts_enabled = st.toggle("Enable Text-to-Speech", value=False)
        uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3'], key="audio_uploader")
        if uploaded_audio:
            transcribed_text = azure_services.transcribe_audio_file(uploaded_audio)
            st.session_state.prompt_from_audio = transcribed_text

def render_chat_messages():
    """Renders chat messages, now with potential image display."""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            # Display the image if it was part of the user's message
            if message["role"] == "user" and "image" in message and message["image"]:
                st.image(message["image"])
            
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "audio" in message and message["audio"]:
                st.audio(message["audio"], format="audio/wav")
# app.py
import streamlit as st
from dotenv import load_dotenv
from src import database, azure_services, ui_components

# --- Page Config and Initialization ---
st.set_page_config(page_title="Azure AI Chat", layout="centered")
load_dotenv()
database.init_db()

# Initialize session state
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "staged_image" not in st.session_state:
    st.session_state.staged_image = None

# --- UI Rendering ---
ui_components.render_sidebar()
st.title("Azure AI Chat Assistant")
ui_components.render_chat_messages()

# --- User Input Handling ---
if prompt := st.chat_input("Ask about the staged image or start a new topic..."):
    
    if st.session_state.current_chat_id is None:
        new_chat_title = prompt[:40] + "..."
        st.session_state.current_chat_id = database.add_chat(new_chat_title)
        st.session_state.messages = []

    # Prepare user message with the staged image if it exists
    user_message = {"role": "user", "content": prompt}
    if st.session_state.staged_image:
        user_message["image"] = st.session_state.staged_image
        st.session_state.staged_image = None # Clear the image after staging it
        
    st.session_state.messages.append(user_message)
    database.add_message(st.session_state.current_chat_id, "user", prompt)

    # Rerun to display the user's message and image immediately
    st.rerun()

# --- AI Response Generation ---
# This block runs only after a user message has been submitted and displayed
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_message_to_process = st.session_state.messages[-1]
    image_to_process = user_message_to_process.get("image")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = azure_services.get_chat_completion(
                st.session_state.messages,
                st.session_state.get("vector_store"),
                image_data=image_to_process
            )
            response_content = st.write_stream(stream)
    
    assistant_message = {"role": "assistant", "content": response_content}
    
    # You have experience with text-to-speech, so this block handles that feature[1].
    if st.session_state.get("tts_enabled", False):
        with st.spinner("Synthesizing audio..."):
            audio_data = azure_services.synthesize_text_to_speech(response_content)
            if audio_data:
                assistant_message["audio"] = audio_data
    
    st.session_state.messages.append(assistant_message)
    # FIX: Corrected variable name from "current_text_chat_id" to "current_chat_id"
    database.add_message(st.session_state.current_chat_id, "assistant", response_content)
    
    # Final rerun to settle the UI
    st.rerun()

# --- Microphone Button Logic ---
if st.button("🎤", key="mic_button", help="Transcribe audio from your microphone"):
    transcribed_text = azure_services.transcribe_audio_from_mic()
    if "Error" not in transcribed_text and transcribed_text:
        st.info(f"🎤 Transcription: {transcribed_text}")
        st.warning("Please copy the text into the chat box to send.")
    else:
        st.error("Failed to transcribe. Please check microphone permissions and try again.")

