import streamlit as st
import os
import tempfile
from pypdf import PdfReader
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests
import uuid
import time
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="ExamGPT: Graph RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables with persistence
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Create a file to store our configuration and processed files
config_path = Path("exam_gpt_config.json")

# Default empty API keys
api_keys = {
    "OPENAI_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "QDRANT_URL": "",
    "QDRANT_API_KEY": ""
}

# Load configuration from file
if config_path.exists():
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            if 'collection_name' in config:
                st.session_state.collection_name = config['collection_name']
            if 'processed_files' in config and isinstance(config['processed_files'], list):
                # Update session state with stored file list
                st.session_state.processed_files = config['processed_files']
            if 'api_keys' in config:
                # Load stored API keys
                for key, value in config['api_keys'].items():
                    api_keys[key] = value  # Update the api_keys dictionary directly
                    st.session_state[key] = value  # Also update session state
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")

# If no saved collection, create a new one
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = f"exam_docs_{uuid.uuid4().hex[:8]}"
    # Save the collection name
    try:
        with open(config_path, "w") as f:
            json.dump({
                "collection_name": st.session_state.collection_name, 
                "processed_files": st.session_state.processed_files,
                "api_keys": api_keys
            }, f)
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")

# Set OpenAI API key
if api_keys["OPENAI_API_KEY"]:
    openai.api_key = api_keys["OPENAI_API_KEY"]

def save_config():
    """Save the current configuration and processed files to disk."""
    try:
        with open(config_path, "w") as f:
            json.dump({
                "collection_name": st.session_state.collection_name,
                "processed_files": st.session_state.processed_files,
                "api_keys": {
                    "OPENAI_API_KEY": api_keys["OPENAI_API_KEY"],
                    "OPENROUTER_API_KEY": api_keys["OPENROUTER_API_KEY"],
                    "QDRANT_URL": api_keys["QDRANT_URL"],
                    "QDRANT_API_KEY": api_keys["QDRANT_API_KEY"]
                }
            }, f)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

# Initialize Qdrant client with retry logic
@st.cache_resource
def get_qdrant_client(max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if api_keys["QDRANT_URL"] and api_keys["QDRANT_API_KEY"]:
                return QdrantClient(url=api_keys["QDRANT_URL"], api_key=api_keys["QDRANT_API_KEY"], timeout=10.0)
            else:
                # Use local Qdrant instance
                return QdrantClient(":memory:", timeout=10.0)
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                st.error(f"Failed to connect to Qdrant after {max_retries} attempts: {str(e)}")
                raise
            time.sleep(1)  # Wait before retrying

# Functions for document processing
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file with error handling and debugging."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        reader = PdfReader(temp_path)
        if len(reader.pages) == 0:
            return "", "PDF has no pages"
            
        text = ""
        page_count = len(reader.pages)
        st.write(f"Extracting text from {pdf_file.name} ({page_count} pages)")
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                # Show first 100 chars of first page for debugging
                if i == 0:
                    preview = page_text[:100] + "..." if len(page_text) > 100 else page_text
                    st.write(f"First page preview: {preview}")
            else:
                st.warning(f"Could not extract text from page {i+1} in {pdf_file.name}")
        
        # Show text statistics
        text = text.strip()
        word_count = len(text.split())
        st.write(f"Extracted {len(text)} characters, approximately {word_count} words")
        
        if not text:
            return "", "Could not extract any text from PDF"
            
        return text, None
    except Exception as e:
        return "", f"Error extracting text: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
  
# Fixed text chunking function
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks with improved boundary detection."""
    chunks = []
    start = 0
    text_length = len(text)
    
    # Reasonable max chunks for a document (e.g., 1 chunk per 100 chars would be extreme)
    max_chunks = max(1, text_length // 100)
    chunk_count = 0
    
    # For debugging
    st.write(f"Text length: {text_length} characters")
    
    while start < text_length and chunk_count < max_chunks:
        # Ensure minimum progress in each iteration
        min_end = start + max(50, chunk_size // 10)  # At least move forward by 10% of chunk_size
        end = min(start + chunk_size, text_length)
        
        # Try to find a good boundary
        if end < text_length:
            # Look for paragraph breaks first (double newline)
            paragraph_break = text.find('\n\n', max(end - 100, 0), min(end + 100, text_length))
            if paragraph_break != -1 and paragraph_break > min_end:
                end = paragraph_break + 2
            else:
                # Look for sentence end (period followed by space or newline)
                sentence_end = -1
                for i in range(end, min(end + 100, text_length)):
                    if i < text_length - 1 and text[i] == '.' and (text[i+1] == ' ' or text[i+1] == '\n'):
                        sentence_end = i + 1
                        break
                
                if sentence_end != -1 and sentence_end > min_end:
                    end = sentence_end
                else:
                    # Fall back to space
                    space = text.find(' ', end)
                    if space != -1 and space < end + 50 and space > min_end:
                        end = space + 1
        
        # Create the chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            chunk_count += 1
            
            # Debug output for the first few chunks
            if chunk_count <= 3:
                st.write(f"Chunk {chunk_count}: {len(chunk)} chars, start={start}, end={end}")
        
        # Always advance start by at least 10% of chunk_size
        next_start = end - overlap
        if next_start <= start + (chunk_size // 10):
            next_start = start + (chunk_size // 10)
        start = next_start
        
    # Add a warning if we hit the max chunks limit
    if chunk_count >= max_chunks:
        st.warning(f"Document chunking stopped at {chunk_count} chunks to prevent processing issues.")
    
    st.write(f"Created {len(chunks)} chunks from {text_length} characters")
    return chunks

# Fixed embedding creation function
def create_embeddings(chunks, max_retries=3):
    """Create embeddings for text chunks with retry logic."""
    embeddings = []
    
    # Process chunks in batches to avoid rate limits
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    st.write(f"Processing {len(chunks)} chunks in {total_batches} batches")
    progress_bar = st.progress(0)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        retry_count = 0
        
        # Update progress bar
        progress_bar.progress(i / len(chunks))
        st.write(f"Processing batch {batch_num} of {total_batches} ({len(batch)} chunks)")
        
        while retry_count < max_retries:
            try:
                response = openai.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                
                for j, embedding_data in enumerate(response.data):
                    embeddings.append({
                        'text': batch[j],
                        'embedding': embedding_data.embedding
                    })
                
                # Pause to avoid rate limits
                if i + batch_size < len(chunks):
                    time.sleep(1)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    st.error(f"Error creating embeddings after {max_retries} retries: {str(e)}")
                    return embeddings  # Return what we have so far
                
                # Exponential backoff
                time.sleep(2 ** retry_count)
    
    progress_bar.progress(1.0)
    return embeddings

def initialize_qdrant_collection(collection_name, vector_size=3072):
    """Initialize a Qdrant collection for storing embeddings."""
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            # Add HNSW index for faster search
            client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,  # Number of connections per layer
                    ef_construct=100  # Controls recall during indexing
                )
            )
        
        return True
    except Exception as e:
        st.error(f"Error initializing Qdrant collection: {str(e)}")
        return False

def get_processed_files_from_qdrant(collection_name):
    """Retrieve the list of processed files from Qdrant collection metadata."""
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            return []
        
        # Search for all unique document names in the collection
        try:
            # Get all points (with limit set very high to get most documents)
            # In a real system, you'd use pagination, but this works for a small number of docs
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter=None,  # No filter means get all
                limit=10000,  # High limit to get most docs
                with_payload=True,
                with_vectors=False  # Don't need vectors
            )[0]
            
            # Extract unique document names
            unique_docs = set()
            for point in results:
                doc_name = point.payload.get('document')
                if doc_name and doc_name != 'unknown':
                    unique_docs.add(doc_name)
            
            return list(unique_docs)
            
        except Exception as e:
            st.warning(f"Error retrieving document list: {str(e)}")
            return []
            
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {str(e)}")
        return []

def sync_processed_files():
    """Sync the list of processed files with what's actually in Qdrant."""
    if not st.session_state.collection_name:
        st.error("No collection name set")
        return False
    
    # Get files from Qdrant
    files_in_qdrant = get_processed_files_from_qdrant(st.session_state.collection_name)
    
    # Update session state
    st.session_state.processed_files = files_in_qdrant
    
    # Save to config file
    save_config()
    
    return True

def store_embeddings_in_qdrant(collection_name, embeddings, metadata):
    """Store embeddings in Qdrant with metadata and retry logic."""
    client = get_qdrant_client()
    points = []
    
    for i, item in enumerate(embeddings):
        point_id = str(uuid.uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=item['embedding'],
                payload={
                    'text': item['text'],
                    'document': metadata.get('filename', 'unknown'),
                    'chunk_index': i,
                    'total_chunks': len(embeddings)
                }
            )
        )
    
    # Store points in batches with retry logic
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    st.error(f"Error storing embeddings in Qdrant after {max_retries} retries: {str(e)}")
                    return False
                time.sleep(1)  # Wait before retrying
    
    return True

def search_similar_chunks(collection_name, query_embedding, limit=5):
    """Search for similar chunks in Qdrant with improved context."""
    client = get_qdrant_client()
    
    try:
        # First, get direct matches
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        # Find related chunks (chunks that come before and after the matched chunks)
        # Group by document
        document_matches = {}
        for result in results:
            doc_name = result.payload.get('document', 'unknown')
            if doc_name not in document_matches:
                document_matches[doc_name] = []
            document_matches[doc_name].append(result)
        
        # Find adjacent chunks for each document
        all_results = list(results)  # Start with direct matches
        for doc_name, doc_results in document_matches.items():
            # Get all chunk indices for this document
            chunk_indices = [r.payload.get('chunk_index') for r in doc_results if r.payload.get('chunk_index') is not None]
            
            if not chunk_indices:
                continue
                
            # Get adjacent indices (both before and after)
            adjacent_indices = set()
            for idx in chunk_indices:
                if idx > 0:
                    adjacent_indices.add(idx - 1)  # Previous chunk
                
                total_chunks = doc_results[0].payload.get('total_chunks', 0)
                if idx < total_chunks - 1:
                    adjacent_indices.add(idx + 1)  # Next chunk
            
            # Remove indices we already have
            adjacent_indices = adjacent_indices - set(chunk_indices)
            
            if not adjacent_indices:
                continue
                
            # Query for adjacent chunks
            try:
                related_results = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document",
                                match=models.MatchValue(value=doc_name)
                            ),
                            models.FieldCondition(
                                key="chunk_index",
                                match=models.MatchAny(any=list(adjacent_indices))
                            )
                        ]
                    ),
                    limit=len(adjacent_indices),
                    with_payload=True
                )[0]
                
                all_results.extend(related_results)
            except Exception as e:
                st.warning(f"Error retrieving related chunks: {str(e)}")
        
        # Convert to standard format and remove duplicates
        unique_results = []
        seen_ids = set()
        
        for point in all_results:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                
                # For direct matches, use the actual score
                if hasattr(point, 'score'):
                    score = point.score
                # For related chunks, calculate a proximity-based score
                else:
                    # Find the closest direct match
                    closest_match_idx = None
                    min_distance = float('inf')
                    point_idx = point.payload.get('chunk_index')
                    
                    for result in results:
                        result_idx = result.payload.get('chunk_index')
                        if result_idx is not None and point_idx is not None:
                            distance = abs(result_idx - point_idx)
                            if distance < min_distance:
                                min_distance = distance
                                closest_match_idx = result_idx
                    
                    # Assign a score based on proximity (closer = higher score)
                    if closest_match_idx is not None:
                        # Score decreases with distance
                        score = max(0.5, 0.9 - 0.1 * min_distance)
                    else:
                        score = 0.5  # Default
                
                unique_results.append({
                    'text': point.payload.get('text', ''),
                    'document': point.payload.get('document', 'unknown'),
                    'chunk_index': point.payload.get('chunk_index'),
                    'score': score
                })
        
        # Sort by score descending
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Sort chunks from the same document by their position
        document_chunks = {}
        for chunk in unique_results:
            doc = chunk['document']
            if doc not in document_chunks:
                document_chunks[doc] = []
            document_chunks[doc].append(chunk)
        
        # Sort each document's chunks by index
        ordered_results = []
        for doc, chunks in document_chunks.items():
            ordered_results.extend(sorted(chunks, key=lambda x: x.get('chunk_index', 0)))
        
        return ordered_results
    
    except Exception as e:
        st.error(f"Error searching Qdrant: {str(e)}")
        return []

def generate_answer(query, context_chunks, model="qwen/qwq-32b"):
    """Generate an answer using OpenRouter API with retry logic."""
    if not api_keys["OPENROUTER_API_KEY"]:
        return "Error: OpenRouter API key not provided", []
    
    # Order chunks by document and position
    ordered_chunks = []
    doc_chunks = {}
    
    for chunk in context_chunks:
        doc = chunk['document']
        if doc not in doc_chunks:
            doc_chunks[doc] = []
        doc_chunks[doc].append(chunk)
    
    # Sort chunks within each document by index
    for doc, chunks in doc_chunks.items():
        ordered_chunks.extend(sorted(chunks, key=lambda x: x.get('chunk_index', 0)))
    
    # Prepare context
    context_parts = []
    for i, chunk in enumerate(ordered_chunks):
        doc_name = chunk['document']
        context_parts.append(f"[Document: {doc_name}]\n{chunk['text']}\n")
    
    context = "\n".join(context_parts)
    
    # Retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_keys['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant helping with exam questions for a HCI College Graduate Course"
                                    "Answer based on the provided context and your understanding of HCI."
                                    "If the information is not in the context, say 'Based on the provided documents, I don't have enough information to answer this question.' "
                                    "Be precise and clear. When quoting from the documents, indicate which document the information comes from."
                        },
                        {
                            "role": "user",
                            "content": f"Here are relevant excerpts from my exam materials:\n\n{context}\n\nBased on these materials and your understanding of HCI, please answer this question: {query}"
                        }
                    ]
                },
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                return answer, ordered_chunks
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    return f"Error: {response.status_code} - {response.text}", []
                
                # Exponential backoff
                time.sleep(2 ** retry_count)
                
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count >= max_retries:
                return "Error: Request to OpenRouter timed out after multiple attempts", []
            
            # Wait longer for timeout issues
            time.sleep(3 ** retry_count)
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return f"Error generating answer after {max_retries} attempts: {str(e)}", []
            
            time.sleep(2 ** retry_count)  # Exponential backoff
    
    return "An unexpected error occurred", []

# Sidebar for API keys and configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Collection information
    st.subheader("Collection Info")
    st.write(f"Current collection: `{st.session_state.collection_name}`")
    
    # Option to reset collection
    if st.button("Create New Collection"):
        new_collection = f"exam_docs_{uuid.uuid4().hex[:8]}"
        st.session_state.collection_name = new_collection
        st.session_state.processed_files = []
        # Save to config file
        save_config()
        st.rerun()  # Use st.rerun() instead of experimental_rerun
        
    # Sync button to check what files are in Qdrant
    if st.button("Sync Files from Database"):
        with st.spinner("Syncing files..."):
            if sync_processed_files():
                st.success(f"Synced {len(st.session_state.processed_files)} files from database")
            else:
                st.error("Failed to sync files from database")
    
    # API Key inputs
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", value=api_keys["OPENAI_API_KEY"], type="password")
    openrouter_key = st.text_input("OpenRouter API Key", value=api_keys["OPENROUTER_API_KEY"], type="password")
    
    # Qdrant configuration
    st.subheader("Qdrant Configuration")
    use_cloud_qdrant = st.checkbox("Use Cloud Qdrant", value=bool(api_keys["QDRANT_URL"]))
    
    if use_cloud_qdrant:
        qdrant_url = st.text_input("Qdrant URL", value=api_keys["QDRANT_URL"])
        qdrant_api_key = st.text_input("Qdrant API Key", value=api_keys["QDRANT_API_KEY"], type="password")
    else:
        st.info("Using in-memory Qdrant instance (data will be lost when app restarts)")
        qdrant_url = ""
        qdrant_api_key = ""
    
    # Update API keys
    if st.button("Update Configuration"):
        # Update API keys dictionary
        api_keys["OPENAI_API_KEY"] = openai_key
        api_keys["OPENROUTER_API_KEY"] = openrouter_key
        api_keys["QDRANT_URL"] = qdrant_url
        api_keys["QDRANT_API_KEY"] = qdrant_api_key
        
        # Also update session state
        st.session_state["OPENAI_API_KEY"] = openai_key
        st.session_state["OPENROUTER_API_KEY"] = openrouter_key
        st.session_state["QDRANT_URL"] = qdrant_url
        st.session_state["QDRANT_API_KEY"] = qdrant_api_key
        
        # Update OpenAI client
        openai.api_key = openai_key
        
        # Save to config file for persistence
        if save_config():
            st.success("Configuration updated and saved!")
        else:
            st.warning("Configuration updated but could not be saved to disk")
    
    # Display processed files
    st.subheader("Processed Files")
    if st.session_state.processed_files:
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    else:
        st.write("No files processed yet")

# Main content
st.title("📚 ExamGPT: Graph RAG for Open Book Exams")
st.markdown("""
This application helps you quickly search and ask questions about your exam documents.
1. Upload PDF documents in the **Upload Documents** tab
2. Ask questions about your documents in the **Ask Questions** tab
""")

# Create tabs
tab1, tab2 = st.tabs(["📄 Upload Documents", "❓ Ask Questions"])

# Upload Documents Tab
with tab1:
    st.header("Upload your PDF documents")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    # Chunking options
    with st.expander("Advanced Options"):
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=700, 
                              help="Number of characters in each chunk (700 recommended for academic material)")
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=175,
                                 help="Number of overlapping characters between chunks (175 recommended)")
        
    # Add info about optimal settings
    with st.expander("Chunking Settings Guide"):
        st.markdown("""
        ### Recommended Settings for Exam Materials
        
        #### Chunk Size
        - **600-800 characters**: Best for most academic content
        - **400-600 characters**: Better for dense technical content with many formulas
        - **800-1200 characters**: Better for narrative content with fewer technical details
        
        #### Chunk Overlap
        - **25% of chunk size**: Ensures concepts don't get split between chunks
        
        #### Number of Chunks to Retrieve
        - **5-7 chunks**: Balanced context for most questions
        - **7-10 chunks**: For complex questions requiring more context
        """)
    
    if uploaded_files:
        if st.button("Process Documents"):
            # Check API keys first
            if not api_keys["OPENAI_API_KEY"]:
                st.error("OpenAI API Key is required for processing documents")
            elif not initialize_qdrant_collection(st.session_state.collection_name):
                st.error("Failed to initialize Qdrant collection")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_files = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_name = uploaded_file.name
                    status_text.text(f"Processing {file_name}...")
                    
                    # Check if file already processed
                    if file_name in st.session_state.processed_files:
                        status_text.text(f"Skipping {file_name} (already processed)")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue
                    
                    # Extract text from PDF
                    text, error = extract_text_from_pdf(uploaded_file)
                    
                    if error:
                        st.error(f"Error processing {file_name}: {error}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue
                    
                    # Split text into chunks
                    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
                    
                    if not chunks:
                        st.warning(f"No text chunks extracted from {file_name}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue
                    
                    # Create embeddings
                    status_text.text(f"Creating embeddings for {file_name}...")
                    embeddings = create_embeddings(chunks)
                    
                    if not embeddings:
                        st.error(f"Failed to create embeddings for {file_name}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue
                    
                    # Store in Qdrant
                    status_text.text(f"Storing embeddings for {file_name}...")
                    if store_embeddings_in_qdrant(
                        st.session_state.collection_name, 
                        embeddings, 
                        {"filename": file_name}
                    ):
                        st.session_state.processed_files.append(file_name)
                        successful_files += 1
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                if successful_files > 0:
                    st.success(f"Successfully processed {successful_files} out of {len(uploaded_files)} documents")
                else:
                    st.error("Failed to process any documents")
                    
                    
# Ask Questions Tab
with tab2:
    st.header("Ask questions about your documents")
    
    if not st.session_state.processed_files:
        st.warning("No documents have been processed. Please upload and process documents first.")
    else:
        query = st.text_area("Enter your question:", height=100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model = st.selectbox(
                "Select model:",
                [
                    "qwen/qwq-32b",
                    "qwen/qwen-max",
                    "google/gemini-2.0-flash-001",
                    "google/gemini-pro",
                    "google/gemini-2.0-flash-thinking-exp:free",
                    "deepseek/deepseek-r1-distill-llama-70b"
                ]
            )
        
        with col2:
            search_limit = st.slider("Number of chunks to retrieve:", min_value=3, max_value=15, value=7)
        
        if st.button("Get Answer") and query:
            if not api_keys["OPENAI_API_KEY"] or not api_keys["OPENROUTER_API_KEY"]:
                st.error("Both OpenAI and OpenRouter API keys are required")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        # Create embedding for query
                        query_embedding_response = openai.embeddings.create(
                            model="text-embedding-3-large",
                            input=query
                        )
                        query_embedding = query_embedding_response.data[0].embedding
                        
                        # Search for similar chunks
                        similar_chunks = search_similar_chunks(
                            st.session_state.collection_name,
                            query_embedding,
                            limit=search_limit
                        )
                        
                        if similar_chunks:
                            # Generate answer
                            answer, sources = generate_answer(query, similar_chunks, model=model)
                            
                            # Display answer
                            st.subheader("Answer:")
                            st.markdown(answer)
                            
                            # Display sources
                            with st.expander("View Source Documents"):
                                # Group sources by document
                                doc_sources = {}
                                for source in sources:
                                    doc = source['document']
                                    if doc not in doc_sources:
                                        doc_sources[doc] = []
                                    doc_sources[doc].append(source)
                                
                                # Display sources by document
                                for doc, chunks in doc_sources.items():
                                    st.markdown(f"### Document: {doc}")

                                    for i, source in enumerate(chunks):
                                        st.markdown(f"**Excerpt {i+1} (Relevance: {source['score']:.2f})**")
                                        st.text_area(f"Excerpt {i+1}", value=source['text'], height=150, disabled=True)

                            st.error("No relevant information found in the documents.")
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("ExamGPT: Graph RAG for Open Book Exams")