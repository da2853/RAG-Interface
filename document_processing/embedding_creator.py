"""
Embedding creator module for ExamGPT application.
Handles creating embeddings for text chunks using OpenAI's API.
"""
import logging
import time
from typing import List, Dict, Any, Optional
import openai
import streamlit as st  # For status updates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EmbeddingCreator:
    """
    Creates embeddings for text chunks using OpenAI's API with
    robust error handling and retry logic.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-large",
                 batch_size: int = 100,
                 max_retries: int = 3,
                 show_status: bool = True):
        """
        Initialize the embedding creator.
        
        Args:
            api_key: OpenAI API key (can be set later)
            model: Embedding model to use
            batch_size: Maximum number of chunks to process in one API call
            max_retries: Maximum number of retry attempts for failed API calls
            show_status: Whether to show embedding status in the Streamlit UI
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.show_status = show_status
        
        if api_key:
            openai.api_key = api_key
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the OpenAI API key.
        
        Args:
            api_key: The OpenAI API key
        """
        if not api_key:
            logger.warning("Empty API key provided")
        
        self.api_key = api_key
        openai.api_key = api_key
    
    def create_embeddings(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Create embeddings for text chunks with retry logic.
        
        Args:
            chunks: List of text chunks to create embeddings for
            
        Returns:
            list: List of dictionaries with 'text' and 'embedding' keys
            
        Raises:
            ValueError: If no API key is set or chunks list is empty
            ConnectionError: If unable to connect to OpenAI API after retries
        """
        if not self.api_key:
            error_msg = "OpenAI API key not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not chunks:
            logger.warning("Empty chunks list provided for embedding creation")
            return []
        
        embeddings = []
        
        # Calculate batches
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches")
        
        if self.show_status:
            st.write(f"Processing {len(chunks)} chunks in {total_batches} batches")
            progress_bar = st.progress(0)
        
        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            
            # Update progress
            if self.show_status:
                progress_bar.progress(i / len(chunks))
                st.write(f"Processing batch {batch_num} of {total_batches} ({len(batch)} chunks)")
            
            # Try to create embeddings with retry logic
            retry_count = 0
            success = False
            
            while retry_count < self.max_retries and not success:
                try:
                    response = openai.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    # Process successful response
                    for j, embedding_data in enumerate(response.data):
                        embeddings.append({
                            'text': batch[j],
                            'embedding': embedding_data.embedding
                        })
                    
                    success = True
                    
                    # Pause to avoid rate limits (only if we have more batches to process)
                    if i + self.batch_size < len(chunks):
                        time.sleep(1)
                    
                except openai.APIConnectionError as e:
                    retry_count += 1
                    logger.warning(f"API connection error on batch {batch_num} (attempt {retry_count}): {str(e)}")
                    if retry_count >= self.max_retries:
                        logger.error(f"Failed to connect to OpenAI API after {self.max_retries} attempts")
                        raise ConnectionError(f"Failed to connect to OpenAI API: {str(e)}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    
                except openai.RateLimitError as e:
                    retry_count += 1
                    logger.warning(f"Rate limit error on batch {batch_num} (attempt {retry_count}): {str(e)}")
                    if retry_count >= self.max_retries:
                        logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                        raise RuntimeError(f"OpenAI API rate limit exceeded: {str(e)}")
                    time.sleep(5 ** retry_count)  # Longer exponential backoff for rate limits
                    
                except openai.APIError as e:
                    retry_count += 1
                    logger.warning(f"API error on batch {batch_num} (attempt {retry_count}): {str(e)}")
                    if retry_count >= self.max_retries:
                        logger.error(f"OpenAI API error after {self.max_retries} attempts")
                        raise RuntimeError(f"OpenAI API error: {str(e)}")
                    time.sleep(2 ** retry_count)
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Unexpected error on batch {batch_num} (attempt {retry_count}): {str(e)}")
                    if retry_count >= self.max_retries:
                        logger.error(f"Unexpected error after {self.max_retries} attempts")
                        raise RuntimeError(f"Error creating embeddings: {str(e)}")
                    time.sleep(2 ** retry_count)
        
        # Complete progress bar
        if self.show_status:
            progress_bar.progress(1.0)
        
        logger.info(f"Successfully created {len(embeddings)} embeddings from {len(chunks)} chunks")
        return embeddings
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create an embedding for a single query string.
        
        Args:
            query: The query text
            
        Returns:
            list: The query embedding vector
            
        Raises:
            ValueError: If no API key is set
            RuntimeError: If unable to create embedding after retries
        """
        if not self.api_key:
            error_msg = "OpenAI API key not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                response = openai.embeddings.create(
                    model=self.model,
                    input=query
                )
                
                return response.data[0].embedding
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error creating query embedding (attempt {retry_count}): {str(e)}")
                
                if retry_count >= self.max_retries:
                    logger.error(f"Failed to create query embedding after {self.max_retries} attempts")
                    raise RuntimeError(f"Failed to create query embedding: {str(e)}")
                
                time.sleep(2 ** retry_count)  # Exponential backoff