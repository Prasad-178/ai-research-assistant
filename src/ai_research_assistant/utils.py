from typing import List, Dict, Any
from pinecone import Pinecone
import openai
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

index = pc.Index(index_name)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[Dict[str, str]]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append({"text": " ".join(current_chunk)})
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append({"text": " ".join(current_chunk)})
    
    return chunks

def index_document(text: str, metadata: Dict[str, Any] = None):
    """Index document chunks in Pinecone"""
    chunks = chunk_text(text)
    index.upsert_records(
      namespace="",
      records=chunks
    )
    

def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Search for relevant documents in Pinecone"""
    query_payload = {
      "inputs": {
        "text": query
      },
      "top_k": top_k,
    }
    
    results = index.search(
      namespace="",
      query=query_payload
    )
    
    return results