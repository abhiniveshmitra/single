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
    # Header
    header_cells = [cell.content for cell in table.cells if cell.row_index == 0]
    md_string += "| " + " | ".join(header_cells) + " |\n"
    md_string += "| " + " | ".join(["---"] * len(header_cells)) + " |\n"
    # Rows
    for row_idx in range(1, table.row_count):
        row_cells = []
        # Ensure cells are added in correct column order
        for col_idx in range(table.column_count):
            cell_found = False
            for cell in table.cells:
                if cell.row_index == row_idx and cell.column_index == col_idx:
                    row_cells.append(cell.content.replace("\n", " ").replace("|", "\|")) # Sanitize content
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
    
    # Read the file bytes into memory once.
    file_bytes = uploaded_file.read()
    
    try:
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
        )
        
        # --- THE RADICAL CHANGE & FIX ---
        # We now explicitly use the 'analyze_request' keyword argument to pass the file bytes.
        # This is the modern, correct way to call the SDK and resolves the "missing 'body'" error.
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", analyze_request=file_bytes, content_type="application/octet-stream"
        )
        result = poller.result()
        st.success("Document analysis complete.")

    except Exception as e:
        # This provides a much cleaner error message to the user for any failure.
        st.error(f"Error during document analysis. This could be a network issue, a problem with your credentials, or an invalid document format. Please check your setup. Details: {e}")
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
        st.warning("No content could be extracted from the document.")
        return False

    st.info(f"Indexing {len(documents_to_upload)} chunks into Azure AI Search...")
    try:
        search_client.upload_documents(documents=documents_to_upload)
        st.success("Document successfully indexed and is now searchable.")
        return True
    except Exception as e:
        st.error(f"Error indexing documents. This could be a network issue or an index configuration problem. Details: {e}")
        return False
