# src/document_processor.py

import os
import uuid
import time
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
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
                    sanitized_content = cell.content.replace("\n", " ").replace("|", "\|")
                    row_cells.append(sanitized_content)
                    cell_found = True
                    break
            if not cell_found:
                row_cells.append("") 
        md_string += "| " + " | ".join(row_cells) + " |\n"
    return md_string

def process_and_index_document(uploaded_file, search_client: SearchClient):
    """
    Orchestrates the process of analyzing a document and indexing its content
    into Azure AI Search using a robust, batched upload strategy.
    """
    doc_intel_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intel_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

    if not all([doc_intel_endpoint, doc_intel_key]):
        st.error("Document Intelligence credentials are not configured in .env file.")
        return False

    with st.spinner(f"Analyzing document '{uploaded_file.name}'..."):
        file_bytes = uploaded_file.read()
        try:
            document_intelligence_client = DocumentIntelligenceClient(
                endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
            )
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", file_bytes, content_type="application/octet-stream"
            )
            result = poller.result()
            st.success("✅ Document analysis complete.")
        except Exception as e:
            st.error(f"Error during document analysis. This could be a network issue, a credential problem, or an invalid document format. Details: {e}")
            return False

    with st.spinner("Preparing and indexing content..."):
        documents_to_upload = []
        if result.paragraphs:
            for para in result.paragraphs:
                documents_to_upload.append({
                    "id": str(uuid.uuid4()), "content": para.content, "source_filename": uploaded_file.name,
                    "page_number": para.bounding_regions[0].page_number if para.bounding_regions else 1, "chunk_type": "paragraph"
                })
        if result.tables:
            for table in result.tables:
                documents_to_upload.append({
                    "id": str(uuid.uuid4()), "content": table_to_markdown(table), "source_filename": uploaded_file.name,
                    "page_number": table.bounding_regions[0].page_number if table.bounding_regions else 1, "chunk_type": "table"
                })
        
        if not documents_to_upload:
            st.warning("⚠️ No content could be extracted from the document.")
            return False

        # --- THE RADICAL CHANGE: BATCHING THE UPLOAD ---
        # Instead of one large upload, we send the data in smaller, more reliable chunks.
        batch_size = 50  # A safe batch size
        total_chunks = len(documents_to_upload)
        
        progress_bar = st.progress(0, text=f"Indexing 0/{total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            batch = documents_to_upload[i:i + batch_size]
            try:
                search_client.upload_documents(documents=batch)
                progress_bar.progress((i + len(batch)) / total_chunks, text=f"Indexing {i + len(batch)}/{total_chunks} chunks...")
                time.sleep(0.5) # A small delay can also help with very strict firewalls
            except Exception as e:
                st.error(f"Error indexing batch starting at chunk {i}. This is likely a persistent network/firewall issue. Please contact your IT support. Details: {e}")
                progress_bar.empty()
                return False
        
        progress_bar.empty()
        st.success(f"✅ Successfully indexed all {total_chunks} chunks. The document is now ready for Q&A.")
        return True
