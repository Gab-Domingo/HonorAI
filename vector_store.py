import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from openai import OpenAI

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def create_document_embeddings(document_text):
    """
    Create embeddings for document chunks using OpenAI embeddings
    """
    # Split the document into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    document_chunks = text_splitter.split_text(document_text)
    
    # Get embeddings for each chunk
    chunk_embeddings = []
    for chunk in document_chunks:
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)
    
    # Create a FAISS index for fast similarity search
    embeddings_array = np.array(chunk_embeddings).astype('float32')
    
    # Get dimension of embeddings
    dimension = embeddings_array.shape[1]
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings_array)
    
    return {
        "index": index,
        "chunks": document_chunks,
        "embeddings": embeddings_array
    }

def get_embedding(text):
    """
    Get embedding vector for text using OpenAI embeddings
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def perform_document_search(query, document_embeddings, full_text=None):
    """
    Search for relevant document chunks based on query
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    query_embedding_array = np.array([query_embedding]).astype('float32')
    
    # Perform similarity search
    index = document_embeddings["index"]
    chunks = document_embeddings["chunks"]
    
    # Get the top 3 most similar chunks
    k = min(3, len(chunks))
    distances, indices = index.search(query_embedding_array, k)
    
    # Get the text of the most relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    
    if full_text and not relevant_chunks:
        # If no relevant chunks found or index is empty, return a portion of the full text
        max_chars = 3000
        return full_text[:max_chars]
    
    # Combine relevant chunks
    context = "\n\n".join(relevant_chunks)
    
    return context

def split_text_for_embeddings(text, chunk_size=1000, chunk_overlap=100):
    """
    Split text into chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)
