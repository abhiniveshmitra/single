# src/document_processor.py

import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    Handles chunking text, creating embeddings, and searching an in-memory FAISS index.
    Now uses a smart index selection logic to handle documents of all sizes.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.text_chunks = []

    def chunk_and_vectorize(self, text: str):
        """Chunks the text and builds the appropriate FAISS index based on data size."""
        # 1. Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.text_chunks = text_splitter.split_text(text)

        # 2. Create embeddings for the chunks
        embeddings = self.embedding_model.embed_documents(self.text_chunks)
        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]

        # --- THE DEFINITIVE FIX: Smart Index Selection ---
        num_chunks = len(self.text_chunks)
        
        # FAISS's IVF index needs at least 39 training points by default.
        # If we have fewer chunks than that, use a simpler, exact search index.
        if num_chunks < 39:
            # Use IndexFlatL2 for small datasets. It's a brute-force search
            # and doesn't require any training.
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
        else:
            # For larger datasets, use the efficient approximate search index.
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, int(np.sqrt(num_chunks))) # A common heuristic for nlist
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Train the index with the embeddings
            self.index.train(embeddings_np)
            self.index.add(embeddings_np)


    def search(self, query: str, k: int = 4):
        """Searches the FAISS index for the most relevant chunks."""
        if not self.index:
            return []
        
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = [self.text_chunks[i] for i in indices[0] if i != -1 and i < len(self.text_chunks)]
        return results
