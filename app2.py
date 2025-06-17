# src/azure_services.py

import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Define the path for the persistent local vector database.
CHROMA_PATH = "chroma_db"

# --- Client Initialization ---

@st.cache_resource
def get_azure_openai_client():
    """Initializes and returns a cached AzureOpenAI client."""
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

# --- NEW: Function to load an existing ChromaDB collection ---

def load_chroma_collection(collection_name: str):
    """
    Loads an existing ChromaDB collection from disk. This allows the app to
    persist the document context across reruns within the same session.
    """
    try:
        # Initialize the embeddings model, which is needed to interact with the collection.
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("ADA_DEPLOYMENT_NAME"),
            openai_api_version="2024-05-01-preview",
        )
        
        # Load the persistent database from disk
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

# --- Modified OpenAI Service ---

def get_chat_completion(messages_from_ui, vector_store: Chroma, image_data=None):
    """
    Generates a chat response, using a local ChromaDB vector store for context.
    Your expertise in building these search systems is reflected in this robust implementation[1].
    """
    client = get_azure_openai_client()
    
    # --- Dynamic System Prompt Logic ---
    context = ""
    # Check if a vector store is available and has been passed.
    if vector_store:
        last_user_message = messages_from_ui[-1]['content']
        # Perform a similarity search on the local ChromaDB to find relevant context[5].
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

    # Prepare the messages for the API call
    api_messages = [{"role": "system", "content": system_prompt}] + \
                   [{"role": msg["role"], "content": msg["content"]} for msg in messages_from_ui]
    
    # The pattern of using Azure OpenAI embeddings for document search is well-established[3][4].
    try:
        return client.chat.completions.create(
            model=os.getenv("GPT4_DEPLOYMENT_NAME"),
            messages=api_messages,
            stream=True,
            temperature=0.7 # A balanced temperature for both factual and general conversation
        )
    except Exception as e:
        st.error(f"Error connecting to Azure OpenAI: {e}")
        # Return an empty iterator to prevent the app from crashing on network errors
        return iter([])

# --- Speech and Other Services (Remain unchanged) ---

def synthesize_text_to_speech(text):
    """Generates speech from text using Azure Speech Services."""
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None
