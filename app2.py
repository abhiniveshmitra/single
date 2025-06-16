# src/azure_services.py
import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# --- Client Initialization (Cached for performance) ---

@st.cache_resource
def get_azure_openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview", # Ensure this is a valid and recent API version
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

@st.cache_resource
def get_speech_config():
    return speechsdk.SpeechConfig(
        subscription=os.getenv("SPEECH_KEY"),
        region=os.getenv("SPEECH_REGION")
    )

@st.cache_resource
def get_computer_vision_client():
    return ComputerVisionClient(
        os.getenv("VISION_ENDPOINT"),
        CognitiveServicesCredentials(os.getenv("VISION_KEY"))
    )

# --- OpenAI Service ---

def get_chat_completion(messages_from_ui, vector_store=None):
    client = get_azure_openai_client()
    system_prompt = "You are a helpful AI assistant. Answer the user's questions. If context from a document is provided, use it to inform your answer."
    
    # RAG: If a vector store is provided, retrieve relevant context
    # Use the last message from the UI data to get the user query
    if vector_store and messages_from_ui and messages_from_ui[-1]['role'] == 'user':
        user_query = messages_from_ui[-1]['content']
        docs = vector_store.similarity_search(user_query, k=4) # k=4 is a common choice
        context = "\n\n".join([doc.page_content for doc in docs])
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENTS ---\n{context}\n--- END OF CONTEXT ---"

    # --- FIX: Prepare a "clean" list of messages for the API ---
    # The API only expects 'role' and 'content' (and optionally 'name' or 'tool_calls').
    # We strip out any other keys (like 'audio') that were added for UI purposes.
    api_messages = []
    for msg in messages_from_ui:
        api_messages.append({"role": msg["role"], "content": msg["content"]}) 

    # Prepend the system prompt to the cleaned message history
    full_api_messages = [{"role": "system", "content": system_prompt}] + api_messages
    
    return client.chat.completions.create(
        model=os.getenv("GPT4_DEPLOYMENT_NAME"),
        messages=full_api_messages, # Send the cleaned list to the API
        stream=True,
        temperature=0.7 # A common setting for balanced creativity and coherence
    )

# --- Speech Services (No changes needed in these functions) ---

def transcribe_audio_from_mic():
    speech_config = get_speech_config()
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Listening... Speak into your microphone.")
    result = recognizer.recognize_once_async().get()
    st.info("Processing complete.")
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Error: Could not recognize speech."

def transcribe_audio_file(audio_file):
    speech_config = get_speech_config()
    # Create an in-memory stream from the uploaded file
    audio_bytes = audio_file.read()
    stream = speechsdk.audio.PushAudioInputStream()
    stream.write(audio_bytes)
    stream.close()
    audio_config = speechsdk.audio.AudioConfig(stream=stream) # Use the PushAudioInputStream
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Transcribing audio file...")
    result = recognizer.recognize_once_async().get()
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Could not transcribe audio."

def synthesize_text_to_speech(text):
    speech_config = get_speech_config()
    # To play the audio in the browser, we don't need a file output (audio_config=None)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None

# --- Computer Vision Service (No changes needed here) ---

def analyze_image(image_stream):
    client = get_computer_vision_client()
    try:
        # Reset stream position if it has been read before
        if hasattr(image_stream, 'seek'):
            image_stream.seek(0)
        description = client.describe_image_in_stream(image_stream)
        
        if hasattr(image_stream, 'seek'):
            image_stream.seek(0) # Reset for the next API call
        tags = client.tag_image_in_stream(image_stream)
        
        caption = "No description generated."
        if description.captions:
            caption = description.captions[0].text

        tag_names = [tag.name for tag in tags.tags]
        
        return {"description": caption, "tags": tag_names}
    except Exception as e:
        return {"error": str(e)}

