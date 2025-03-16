"""
Qdrant manager module for ExamGPT application.
Handles interactions with the Qdrant vector database.
"""
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
import streamlit as st  # For status updates
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages interactions with the Qdrant vector database for storing
    and retrieving document embeddings.
    """
    
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: float = 10.0,
                 vector_size: int = 3072,
                 max_retries: int = 3,
                 show_status: bool = True):
        """
        Initialize the Qdrant manager.
        
        Args:
            url: Qdrant server URL (uses in-memory if None)
            api_key: Qdrant API key (not needed for in-memory)
            timeout: Timeout for Qdrant operations in seconds
            vector_size: Size of the embedding vectors
            max_retries: Maximum retry attempts for operations
            show_status: Whether to show operation status in the Streamlit UI
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.vector_size = vector_size
        self.max_retries = max_retries
        self.show_status = show_status
        self._client = None
    
    @property
    def client(self) -> QdrantClient:
        """
        Get or initialize the Qdrant client.
        
        Returns:
            QdrantClient: The Qdrant client instance
            
        Raises:
            ConnectionError: If unable to connect to Qdrant after retries
        """
        if self._client is None:
            retry_count = 0
            
            while retry_count < self.max_retries:
                try:
                    if self.url and self.api_key:
                        self._client = QdrantClient(
                            url=self.url,
                            api_key=self.api_key,
                            timeout=self.timeout
                        )
                        logger.info(f"Connected to Qdrant at {self.url}")
                    else:
                        # Use in-memory instance
                        self._client = QdrantClient(":memory:", timeout=self.timeout)
                        logger.info("Using in-memory Qdrant instance")
                    
                    break  # Successfully connected
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error connecting to Qdrant (attempt {retry_count}): {str(e)}")
                    
                    if retry_count >= self.max_retries:
                        logger.error(f"Failed to connect to Qdrant after {self.max_retries} attempts")
                        raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")
                    
                    time.sleep(2 ** retry_count)  # Exponential backoff
        
        return self._client
    
    def initialize_collection(self, collection_name: str) -> bool:
        """
        Initialize a Qdrant collection for storing embeddings.
        Creates the collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Add HNSW index for faster search
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,  # Number of connections per layer
                    ef_construct=100  # Controls recall during indexing
                )
            )
            
            logger.info(f"Created new collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            if self.show_status:
                st.error(f"Error initializing Qdrant collection: {str(e)}")
            return False
    
    def get_processed_files(self, collection_name: str) -> List[str]:
        """
        Retrieve the list of processed files from Qdrant collection metadata.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            list: List of processed file names
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Search for all unique document names in the collection
            try:
                # Get all points (with limit set high to get most documents)
                results = self.client.scroll(
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
                
                logger.info(f"Found {len(unique_docs)} processed files in collection {collection_name}")
                return list(unique_docs)
                
            except Exception as e:
                logger.warning(f"Error retrieving document list: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            return []
    
    def store_embeddings(self, 
                         collection_name: str, 
                         embeddings: List[Dict[str, Any]], 
                         metadata: Dict[str, Any]) -> bool:
        """
        Store embeddings in Qdrant with metadata and retry logic.
        
        Args:
            collection_name: Name of the collection
            embeddings: List of dictionaries with 'text' and 'embedding' keys
            metadata: Additional metadata for the embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not embeddings:
            logger.warning("No embeddings to store")
            return False
        
        points = []
        
        # Prepare points for Qdrant
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
        success = True
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            retry_count = 0
            batch_success = False
            
            while retry_count < self.max_retries and not batch_success:
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    batch_success = True
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error storing batch {i//batch_size + 1} (attempt {retry_count}): {str(e)}")
                    
                    if retry_count >= self.max_retries:
                        logger.error(f"Failed to store batch after {self.max_retries} attempts")
                        success = False
                    
                    time.sleep(2 ** retry_count)  # Exponential backoff
            
            # Log progress for large batches
            if len(points) > batch_size and i % (5 * batch_size) == 0:
                logger.info(f"Stored {i + len(batch)}/{len(points)} points")
        
        if success:
            logger.info(f"Successfully stored {len(embeddings)} embeddings for {metadata.get('filename', 'unknown')}")
        
        return success
    
    def search_similar_chunks(self, 
                             collection_name: str, 
                             query_embedding: List[float], 
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant with improved context.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            list: List of similar chunks with metadata
        """
        try:
            # First, get direct matches
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
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
                chunk_indices = [
                    r.payload.get('chunk_index') 
                    for r in doc_results 
                    if r.payload.get('chunk_index') is not None
                ]
                
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
                    related_results = self.client.scroll(
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
                    logger.warning(f"Error retrieving related chunks: {str(e)}")
            
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
            
            logger.info(f"Found {len(ordered_results)} relevant chunks ({len(results)} direct matches)")
            return ordered_results
        
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            if self.show_status:
                st.error(f"Error searching Qdrant: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False