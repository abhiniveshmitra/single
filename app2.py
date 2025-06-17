# src/azure_services.py

import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # The new import for local embeddings

# Define the constants for the local vector database and embedding model
CHROMA_PATH = "chroma_db"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A fast and effective local model

# --- Client Initialization (No changes here) ---

@st.cache_resource
def get_azure_openai_client():
    """Initializes and returns a cached AzureOpenAI client for chat."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

@st.cache_resource
def get_speech_config():
    """Initializes and returns a cached Azure Speech service configuration."""
    return speechsdk.SpeechConfig(
        subscription=os.getenv("SPEECH_KEY"),
        region=os.getenv("SPEECH_REGION")
    )

# --- Function to load ChromaDB using LOCAL embeddings ---

def load_chroma_collection(collection_name: str):
    """
    Loads an existing ChromaDB collection from disk using the local embedding model.
    """
    try:
        # --- THE RADICAL CHANGE ---
        # We now instantiate the local HuggingFace model to interact with the database.
        # This ensures consistency and avoids network calls during loading.
        embeddings_model = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings_model,
            collection_name=collection_name
        )
        return db
    except Exception as e:
        # This can happen if the collection directory doesn't exist yet.
        st.info(f"Chroma collection '{collection_name}' not found. It will be created when a document is processed.")
        return None

# --- OpenAI Service using the Local Vector Store ---

def get_chat_completion(messages_from_ui, vector_store: Chroma, image_data=None):
    """
    Generates a chat response, using a local ChromaDB vector store for context.
    The logic here remains the same as it correctly handles the dynamic prompt.
    """
    client = get_azure_openai_client()
    
    # Dynamic System Prompt Logic
    context = ""
    if vector_store:
        last_user_message = messages_from_ui[-1]['content']
        # Perform a similarity search on the local ChromaDB to find relevant context.
        results = vector_store.similarity_search(last_user_message, k=4)
        if results:
            context = "\n\n".join([doc.page_content for doc in results])

    if context:
        # If we found relevant context, instruct the bot to be a document expert.
        system_prompt = (
            "You are an expert AI assistant for document analysis. Answer the user's questions based ONLY on the provided context from the document. "
            "If the answer is not in the context, clearly state that the document does not contain the answer. "
            "Do not use any outside knowledge."
        )
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENT ---\n{context}\n--- END OF CONTEXT ---"
    else:
        # If no context was found, instruct the bot to be a general-purpose assistant.
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question to the best of your ability. "
            "If you are asked about a specific document, inform the user that no document has been processed "
            "or no relevant information was found."
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

# --- Speech Services (No changes here) ---

def synthesize_text_to_speech(text):
    """Generates speech from text using Azure Speech Services."""
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None
