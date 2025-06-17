# src/document_processor.py

import os
import uuid
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient

def table_to_markdown(table):
    """Converts a DocumentIntelligence table object to a Markdown string for better searchability."""
    md_string = ""
    header_cells = [cell.content for cell in table.cells if cell.row_index == 0]
    md_string += "| " + " | ".join(header_cells) + " |\n"
    md_string += "| " + " | ".join(["---"] * len(header_cells)) + " |\n"
    for row_idx in range(1, table.row_count):
        row_cells = []
        for col_idx in range(table.column_count):
            cell_found = False
            for cell in table.cells:
                if cell.row_index == row_idx and cell.column_index == col_idx:
                    row_cells.append(cell.content)
                    cell_found = True
                    break
            if not cell_found:
                row_cells.append("")
        md_string += "| " + " | ".join(row_cells) + " |\n"
    return md_string

def process_and_index_document(uploaded_file, search_client: SearchClient):
    """
    Orchestrates the process of analyzing a document with Document Intelligence
    and indexing its content into Azure AI Search, with robust error handling.
    """
    doc_intel_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intel_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

    if not all([doc_intel_endpoint, doc_intel_key]):
        st.error("Document Intelligence credentials are not configured in .env file.")
        return False

    st.info(f"Analyzing document '{uploaded_file.name}' with Document Intelligence...")
    
    try:
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
        )
        
        # FIX #1: Correctly use the 'analyze_request' keyword argument to pass the file bytes.
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", analyze_request=uploaded_file.read(), content_type="application/octet-stream"
        )
        result = poller.result()
        st.success("Document analysis complete.")
    except Exception as e:
        # FIX #2: Catch any error (including network/EOF errors) and show a clean message.
        st.error(f"Error during document analysis. This could be a network issue or an invalid document format. Details: {e}")
        return False

    st.info("Preparing content for search index...")
    documents_to_upload = []
    
    if result.paragraphs:
        for para in result.paragraphs:
            documents_to_upload.append({
                "id": str(uuid.uuid4()),
                "content": para.content,
                "source_filename": uploaded_file.name,
                "page_number": para.bounding_regions[0].page_number if para.bounding_regions else 1,
                "chunk_type": "paragraph"
            })

    if result.tables:
        for table in result.tables:
            documents_to_upload.append({
                "id": str(uuid.uuid4()),
                "content": table_to_markdown(table),
                "source_filename": uploaded_file.name,
                "page_number": table.bounding_regions[0].page_number if table.bounding_regions else 1,
                "chunk_type": "table"
            })
    
    if not documents_to_upload:
        st.warning("No content was extracted from the document.")
        return False

    st.info(f"Indexing {len(documents_to_upload)} chunks into Azure AI Search...")
    try:
        # FIX #2 (cont.): Add error handling for the search upload process as well.
        search_client.upload_documents(documents=documents_to_upload)
        st.success("Document successfully indexed and is now searchable.")
        return True
    except Exception as e:
        st.error(f"Error indexing documents. This could be a network issue (like an EOF error) or an index configuration problem. Details: {e}")
        return False
# src/azure_services.py

import os
import base64
import streamlit as st
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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
def get_search_client():
    try:
        endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        if not all([endpoint, key, index_name]):
            st.error("Azure AI Search credentials not fully configured in the .env file.")
            return None
        return SearchClient(endpoint, index_name, AzureKeyCredential(key))
    except Exception as e:
        st.error(f"Failed to initialize Azure AI Search client: {e}")
        return None

@st.cache_resource
def get_speech_config():
    return speechsdk.SpeechConfig(
        subscription=os.getenv("SPEECH_KEY"),
        region=os.getenv("SPEECH_REGION")
    )

# --- Azure AI Search Querying with Error Handling ---
def get_context_from_azure_search(query: str, search_client: SearchClient):
    """Performs a hybrid search and returns the context. Handles network errors gracefully."""
    vector_query = VectorizedQuery(vector=[], k_nearest_neighbors=3, fields="content_vector")
    try:
        # FIX #2 (cont.): Wrap the search call in a try...except block.
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name='default',
            top=5
        )
        context = ""
        for result in results:
            context += f"\n[SOURCE: file '{result['source_filename']}', page {result['page_number']}]\n{result['content']}\n"
        return context
    except Exception as e:
        # This will catch the EOFError and display a clean message instead of crashing.
        st.error(f"Could not connect to Azure AI Search. Please check your network/firewall settings. Details: {e}")
        return ""

# --- OpenAI Service with Dynamic Prompting ---
def get_chat_completion(messages_from_ui, search_client: SearchClient, image_data=None):
    client = get_azure_openai_client()
    
    last_user_message = messages_from_ui[-1]['content']
    context = get_context_from_azure_search(last_user_message, search_client)
    
    if context:
        system_prompt = (
            "You are an expert AI assistant for document analysis. Answer the user's questions based ONLY on the provided context. "
            "If the answer is in the context, cite your sources using the [SOURCE: 'filename', page X] format. "
            "If the answer is not in the context, say 'I cannot answer this question based on the provided document.'"
        )
        system_prompt += f"\n\n--- CONTEXT FROM DOCUMENT ---\n{context}\n--- END OF CONTEXT ---"
    else:
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question to the best of your ability. "
            "If you are asked a question about a specific document, inform the user that no document has been processed "
            "or no relevant information was found."
        )

    api_messages = [{"role": "system", "content": system_prompt}] + \
                   [{"role": msg["role"], "content": msg["content"]} for msg in messages_from_ui]
    
    try:
        # FIX #2 (cont.): Add final error handling around the OpenAI call itself.
        return client.chat.completions.create(
            model=os.getenv("GPT4_DEPLOYMENT_NAME"),
            messages=api_messages,
            stream=True,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Error connecting to Azure OpenAI. Please check your network/firewall settings. Details: {e}")
        # Return an empty stream or a specific error message if the call fails.
        # This is an advanced technique; for now, the error message is sufficient.
        return iter([])


# --- Speech Services ---
def synthesize_text_to_speech(text):
    speech_config = get_speech_config()
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None

