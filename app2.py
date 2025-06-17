# src/document_processor.py

import os
import uuid
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from light_embed.embedding import Embedding # THE DEFINITIVE FIX: Correct import statement

CHROMA_PATH = "chroma_db"

def process_and_index_document(uploaded_file, collection_name: str):
    """
    Analyzes a document, then chunks and embeds its content using the
    lightweight and dependency-free LightEmbed library.
    """
    # --- Document Intelligence analysis part remains the same ---
    doc_intel_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intel_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

    with st.spinner(f"Analyzing document '{uploaded_file.name}'..."):
        try:
            document_intelligence_client = DocumentIntelligenceClient(
                endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
            )
            file_bytes = uploaded_file.read()
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", file_bytes, content_type="application/octet-stream"
            )
            result = poller.result()
        except Exception as e:
            st.error(f"Error during document analysis: {e}")
            return None

    full_content = ""
    if result.paragraphs:
        for para in result.paragraphs:
            full_content += para.content + "\n"
    if not full_content:
        st.warning("⚠️ No text content extracted.")
        return None

    # --- Text Splitting remains the same ---
    with st.spinner("Preparing content for local embedding..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=full_content)
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    # --- Using LightEmbed with the correct class name ---
    with st.spinner("Creating embeddings locally with LightEmbed..."):
        try:
            # THE DEFINITIVE FIX: Instantiate the correct 'Embedding' class
            embedding_model = Embedding()
            
            vectors = embedding_model.embed(chunks)

            db = Chroma(
                collection_name=collection_name,
                persist_directory=CHROMA_PATH
            )
            db.add_embeddings(
                ids=chunk_ids,
                embeddings=vectors,
                documents=chunks
            )
            st.success("✅ Document successfully indexed using local LightEmbed engine.")
            return db
        except Exception as e:
            st.error(f"Error creating local vector store with LightEmbed: {e}")
            return None
# src/azure_services.py

import os
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langchain_community.vectorstores import Chroma
from light_embed.embedding import Embedding # THE DEFINITIVE FIX: Correct import statement

CHROMA_PATH = "chroma_db"

# --- Client Initialization and Model Caching ---
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

@st.cache_resource
def get_embedding_model():
    """Loads the LightEmbed model once and caches it."""
    # THE DEFINITIVE FIX: Instantiate the correct 'Embedding' class
    return Embedding()

# --- Function to Load ChromaDB ---
def load_chroma_collection(collection_name: str):
    """Loads an existing ChromaDB collection from disk."""
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            collection_name=collection_name
        )
        return db
    except Exception as e:
        st.info(f"Chroma collection '{collection_name}' not found. It will be created when a document is processed.")
        return None

# --- Chat Completion Function (Refactored for manual query embedding) ---
def get_chat_completion(messages_from_ui, vector_store: Chroma, image_data=None):
    client = get_azure_openai_client()
    context = ""
    if vector_store:
        last_user_message = messages_from_ui[-1]['content']
        
        embedding_model = get_embedding_model()
        query_vector = embedding_model.embed(last_user_message)[0]
        
        results = vector_store.similarity_search_by_vector(embedding=query_vector, k=4)
        
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

