# src/azure_services.py

import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

CHROMA_PATH = "chroma_db"

# --- THE RADICAL CHANGE: CACHED MODEL LOADING ---
@st.cache_resource
def get_embedding_model():
    """
    Loads the AzureOpenAIEmbeddings model once and caches it as a global resource.
    This is the definitive solution to prevent memory-related crashes.
    """
    st.info("Initializing Azure OpenAI embedding model... (This happens only once per session)")
    # On the first run, this will connect to Azure and set up the model object.
    # On every subsequent rerun, Streamlit will return the cached object instantly.
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
        openai_api_version="2024-05-01-preview",
    )


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

# --- Updated Function to Load ChromaDB ---
def load_chroma_collection(collection_name: str):
    """Loads an existing ChromaDB collection using the cached embedding model."""
    try:
        # Use the cached function to get the model object
        embeddings_model = get_embedding_model()
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings_model,
            collection_name=collection_name
        )
        return db
    except Exception as e:
        st.info(f"Chroma collection '{collection_name}' not found.")
        return None

# --- Chat Completion Function (No changes to its internal logic) ---
def get_chat_completion(messages_from_ui, vector_store: Chroma, image_data=None):
    client = get_azure_openai_client()
    context = ""
    if vector_store:
        last_user_message = messages_from_ui[-1]['content']
        results = vector_store.similarity_search(last_user_message, k=4)
        if results:
            context = "\n\n".join([doc.page_content for doc in results])
    if context:
        system_prompt = (
            "You are an expert AI assistant for document analysis. Answer the user's questions based ONLY on the provided context. "
            "If the answer is not in the context, state that the document does not contain the answer."
        )
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENT ---\n{context}\n--- END OF CONTEXT ---"
    else:
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question to the best of your ability. "
            "If asked about a document, say that no document has been processed or no relevant information was found."
        )
    api_messages = [{"role": "system", "content": system_prompt}] + \
                   [{"role": msg["role"], "content": msg["content"]} for msg in messages_from_ui]
    try:
        return client.chat.completions.create(
            model=os.getenv("GPT4_DEPLOYMENT_NAME"),
            messages=api_messages,
            stream=True,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Error connecting to Azure OpenAI: {e}")
        return iter([])

# --- Speech Services (No changes) ---
def synthesize_text_to_speech(text):
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None
